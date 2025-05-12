#!/usr/bin/env python3
"""
Script para generar predicciones en Galapagar usando los últimos 72h de datos de estaciones externas:
  - pontevedra, salamanca, huelva, ciudadreal, valencia, almeria, creus, Santander, Segovia, Guadalajara, Burgos
 1) Descargar y actualizar históricos horarios de AEMET para cada estación (excepto Galapagar)
 2) Preprocesar datos (interpolación, variables temporales, normalización con stats de entrenamiento)
 3) Generar pronóstico hora a hora para las próximas 24h con ForecastNet
 4) Guardar resultados en CSV y gráficos de predicción
No se comparan con datos reales de Galapagar en este script.
"""
import os
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from io import StringIO
import math
from datetime import timedelta
from functools import reduce

# -------------------------
# Estaciones predictoras y URLs AEMET
# -------------------------
station_urls = {
    'pontevedra':  'https://www.aemet.es/es/eltiempo/observacion/ultimosdatos_1484C_datos-horarios.csv?k=gal&l=1484C&datos=det&w=0&f=temperatura&x=',
    'salamanca':   'https://www.aemet.es/es/eltiempo/observacion/ultimosdatos_2870_datos-horarios.csv?k=cle&l=2870&datos=det&w=0&f=temperatura&x=',
    'huelva':      'https://www.aemet.es/es/eltiempo/observacion/ultimosdatos_4642E_datos-horarios.csv?k=and&l=4642E&datos=det&w=0&f=temperatura&x=',
    'ciudadreal':  'https://www.aemet.es/es/eltiempo/observacion/ultimosdatos_4121_datos-horarios.csv?k=clm&l=4121&datos=det&w=0&f=temperatura&x=',
    'valencia':    'https://www.aemet.es/es/eltiempo/observacion/ultimosdatos_8414A_datos-horarios.csv?k=val&l=8414A&datos=det&w=0&f=temperatura&x=',
    'almeria':     'https://www.aemet.es/es/eltiempo/observacion/ultimosdatos_6325O_datos-horarios.csv?k=and&l=6325O&datos=det&w=0&f=temperatura&x=',
    'creus':       'https://www.aemet.es/es/eltiempo/observacion/ultimosdatos_0433D_datos-horarios.csv?k=cat&l=0433D&datos=det&w=0&f=temperatura&x=h24',
    'Santander':   'https://www.aemet.es/es/eltiempo/observacion/ultimosdatos_1111X_datos-horarios.csv?k=can&l=1111X&datos=det&w=0&f=temperatura&x=h24',
    'Segovia':     'https://www.aemet.es/es/eltiempo/observacion/ultimosdatos_2465_datos-horarios.csv?k=cle&l=2465&datos=det&w=0&f=temperatura&x=h24',
    'Guadalajara': 'https://www.aemet.es/es/eltiempo/observacion/ultimosdatos_3168D_datos-horarios.csv?k=clm&l=3168D&datos=det&w=0&f=temperatura&x=h24',
    'Burgos':      'https://www.aemet.es/es/eltiempo/observacion/ultimosdatos_2331_datos-horarios.csv?k=cle&l=2331&datos=det&w=0&f=temperatura&x=h24'
}

# Columnas AEMET y mapeo a stats
numeric_cols = ['Temperatura (ºC)', 'Humedad (%)', 'Precipitación (mm)']
time_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
feature_map = {
    'Temperatura (ºC)': 'temperature_2m (°C)',
    'Humedad (%)': 'relative_humidity_2m (%)',
    'Precipitación (mm)': 'rain (mm)'
}

# -------------------------
# Descarga y parseo CSV AEMET
# -------------------------
def fetch_csv(url):
    resp = requests.get(url); resp.raise_for_status()
    lines = resp.text.splitlines()
    for i, l in enumerate(lines):
        if l.startswith('"Fecha y hora oficial"'):
            return pd.read_csv(StringIO("\n".join(lines[i:])), sep=',', quotechar='"',
                               parse_dates=['Fecha y hora oficial'], dayfirst=True)
    raise RuntimeError("Cabecera no encontrada en datos AEMET.")

# -------------------------
# Actualizar histórico de cada estación
# -------------------------
def update_history(st):
    fn = f"{st}.csv"
    df_new = fetch_csv(station_urls[st]).set_index('Fecha y hora oficial')
    if os.path.exists(fn):
        df_hist = pd.read_csv(fn, parse_dates=['datetime'], index_col='datetime')
    else:
        df_hist = pd.DataFrame()
    df = pd.concat([df_hist, df_new[numeric_cols]], axis=0)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq='h'))
    df[numeric_cols] = df[numeric_cols].interpolate(method='time')
    df['datetime'] = df.index
    df.to_csv(fn, index=False, float_format='%.1f')
    return fn

# -------------------------
# Variables temporales
# -------------------------
def add_time_feats(df):
    t = df['datetime']
    df['hour_sin'] = np.sin(2*np.pi*t.dt.hour/24)
    df['hour_cos'] = np.cos(2*np.pi*t.dt.hour/24)
    df['dow_sin'] = np.sin(2*np.pi*t.dt.dayofweek/7)
    df['dow_cos'] = np.cos(2*np.pi*t.dt.dayofweek/7)
    return df

# -------------------------
# ForecastNet LSTM
# -------------------------
class ForecastNet(nn.Module):
    def __init__(self, in_dim, hid, nl, steps, od, drop=0.2):
        super().__init__()
        self.encoder = nn.LSTM(in_dim, hid, nl, batch_first=True, dropout=drop)
        self.decoder = nn.LSTM(od, hid, nl, batch_first=True, dropout=drop)
        self.fc = nn.Linear(hid, od)
        self.proj = nn.Linear(od, in_dim)
        self.steps = steps
    def forward(self, x):
        _, (h, c) = self.encoder(x)
        inp = x[:, -1:, :2]
        outs = []
        for _ in range(self.steps):
            d = self.proj(inp)
            o, (h, c) = self.decoder(d, (h, c))
            p = self.fc(o)
            outs.append(p)
            inp = p
        return torch.cat(outs, 1)

# -------------------------
# Pipeline config
# -------------------------
INPUT_WINDOW = 24
OUTPUT_WINDOW = 24
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------
# Main
# -------------------------
def main():
    # 1) actualizar históricos
    for st in station_urls:
        print(f"Histórico {st} ->", update_history(st))

    # 2) cargar y unir datos
    dfs = []
    for st in station_urls:
        df = pd.read_csv(f"{st}.csv", parse_dates=['datetime'])
        df = add_time_feats(df)
        df.set_index('datetime', inplace=True)
        df = df[numeric_cols + time_cols]
        df.columns = [f"{st}_{c}" for c in numeric_cols + time_cols]
        dfs.append(df)
    merged = reduce(lambda a, b: a.join(b, how='inner'), dfs).sort_index()

    # 3) normalizar
    stats = pd.read_csv('model/normalization_stats.csv', index_col=0)
    stats.columns = ['mean', 'std']
    merged_norm = merged.copy()
    for col in merged.columns:
        feat = col.split('_', 1)[1]
        if feat in feature_map:
            key = feature_map[feat]
            if key in stats.index:
                m, s = stats.at[key, 'mean'], stats.at[key, 'std']
                merged_norm[col] = (merged[col] - m) / s
    data = merged_norm.values

    # 4) comprobar suficiente historial
    if data.shape[0] < INPUT_WINDOW:
        print(f"No hay suficientes datos para formar una ventana de {INPUT_WINDOW} horas.")
        return

    # 5) preparar última ventana y predecir próximas 24h
    last_window = torch.tensor(data[-INPUT_WINDOW:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    state = torch.load('model/forecast_model.pt', map_location=DEVICE)
    input_dim = last_window.shape[2]
    model = ForecastNet(input_dim, 64, 2, OUTPUT_WINDOW, 2).to(DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()
    with torch.no_grad():
        pred_seq = model(last_window).cpu().numpy().squeeze(0)

    # 6) compilar pronóstico
    last_time = merged.index[-1]
    results = []
    for h in range(1, OUTPUT_WINDOW+1):
        tp_norm, rp_norm = pred_seq[h-1]
        t_pred = last_time + timedelta(hours=h)
        m_t, s_t = stats.at['temperature_2m (°C)', 'mean'], stats.at['temperature_2m (°C)', 'std']
        m_r, s_r = stats.at['rain (mm)', 'mean'], stats.at['rain (mm)', 'std']
        tp = tp_norm * s_t + m_t
        rp = rp_norm * s_r + m_r
        results.append({'time': t_pred, 'horizon_h': h, 'pred_temp': tp, 'pred_rain': rp})
    dfres = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    dfres.to_csv('results/next24h_predictions.csv', index=False)

    # 7) graficar
    plt.figure(); plt.plot(dfres.time, dfres.pred_temp, marker='o'); plt.title('Pronóstico Temp 24h'); plt.tight_layout(); plt.savefig('results/next24h_temp.png')
    plt.figure(); plt.plot(dfres.time, dfres.pred_rain, marker='o'); plt.title('Pronóstico Rain 24h'); plt.tight_layout(); plt.savefig('results/next24h_rain.png')
    print('Pronóstico hora a hora generado.')

if __name__ == '__main__':
    main()
