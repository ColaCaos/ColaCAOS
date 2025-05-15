#!/usr/bin/env python3
"""
Predice3.py

Ejecuta cada hora y genera:
  - results/next24h_predictions.csv: histórico acumulado de predicciones hora a hora para próximas 24h, sin duplicados ni huecos.
  - results/next24h_temp.png y next24h_rain.png: gráficos de las últimas 24h de predicción.
  - results/pivot_pred_temp.csv: tabla acumulada de temperaturas (index=run_time, columns=1–24).
  - results/pivot_pred_rain.csv: tabla acumulada de lluvias.

Basado en Predice2.py con pivot acumulado.
"""
import os
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from io import StringIO
from datetime import timedelta
from functools import reduce

# -------------------------
# Configuración estaciones
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
    'Burgos':      'https://www.aemet.es/es/eltiempo/observacion/ultimosdatos_2331_datos-horarios.csv?k=cle&l=2331&datos=det&w=0&f=temperatura&x=h24',
    'galapagar':   'https://www.aemet.es/es/eltiempo/observacion/ultimosdatos_3272M_datos-horarios.csv?k=mad&l=3272M&datos=det&w=0&f=temperatura&x='
}

# -------------------------
# Features y mapeo
# -------------------------
numeric_cols = [
    'Temperatura (ºC)',
    'Humedad (%)',
    'Precipitación (mm)',
    'Presión (hPa)',
    'Velocidad del viento (km/h)',
    'Dirección del viento'
]
cat_cols = []
time_feats = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']

dir2deg = {
    'Norte': 0, 'Nornoreste': 22.5, 'Noreste': 45, 'Estenoreste': 67.5,
    'Este': 90, 'Estesureste': 112.5, 'Sureste': 135, 'Sursureste': 157.5,
    'Sur': 180, 'Sursuroeste': 202.5, 'Suroeste': 225, 'Oestesuroeste': 247.5,
    'Oeste': 270, 'Oestonoroeste': 292.5, 'Noroeste': 315, 'Nornoroeste': 337.5
}
feature_map = {
    'Temperatura (ºC)': 'temperature_2m (°C)',
    'Humedad (%)': 'relative_humidity_2m (%)',
    'Precipitación (mm)': 'rain (mm)',
    'Presión (hPa)': 'surface_pressure (hPa)',
    'Velocidad del viento (km/h)': 'wind_speed_10m (km/h)'
}

INPUT_WINDOW = 24
OUTPUT_WINDOW = 24
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------
# Funciones utilitarias
# -------------------------
def fetch_csv(url):
    r = requests.get(url)
    r.raise_for_status()
    lines = r.text.splitlines()
    for i, l in enumerate(lines):
        if l.startswith('"Fecha y hora oficial"'):
            return pd.read_csv(
                StringIO("\n".join(lines[i:])), sep=',', quotechar='"',
                parse_dates=['Fecha y hora oficial'], dayfirst=True
            )
    raise RuntimeError('Cabecera no encontrada')

def update_history(st):
    fn = f"{st}.csv"
    df_new = fetch_csv(station_urls[st]).set_index('Fecha y hora oficial')
    if os.path.exists(fn):
        df_hist = pd.read_csv(fn, parse_dates=['datetime'], index_col='datetime')
    else:
        df_hist = pd.DataFrame()
    df = pd.concat([df_hist, df_new[numeric_cols]])
    df = df[~df.index.duplicated()].sort_index()
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq='h'))
    df[numeric_cols] = df[numeric_cols].interpolate(method='time')
    df['Dirección del viento'] = df['Dirección del viento'].map(dir2deg).ffill().bfill()
    df['datetime'] = df.index
    df.to_csv(fn, index=False, float_format='%.1f')
    return fn

def add_time_feats(df):
    t = df['datetime']
    df['hour_sin'] = np.sin(2 * np.pi * t.dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * t.dt.hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * t.dt.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * t.dt.dayofweek / 7)
    return df

class ForecastNet(nn.Module):
    def __init__(self, in_dim, hid, nl, steps, od, drop=0.2):
        super().__init__()
        self.encoder = nn.LSTM(in_dim, hid, nl, batch_first=True, dropout=drop)
        self.decoder = nn.LSTM(in_dim, hid, nl, batch_first=True, dropout=drop)
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
# Pivot helper
# -------------------------
def append_and_save(row, fn):
    # Asegurar directorio
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    if os.path.exists(fn):
        df_prev = pd.read_csv(fn)
        if 'run_time' not in df_prev.columns and 'time' in df_prev.columns:
            df_prev = df_prev.rename(columns={'time': 'run_time'})
        df_prev['run_time'] = pd.to_datetime(df_prev['run_time'])
        df_prev = df_prev.set_index('run_time')
        df_new = pd.DataFrame([row]).set_index('run_time')
        df_comb = pd.concat([df_prev, df_new])
        df_comb = df_comb[~df_comb.index.duplicated(keep='last')].sort_index()
    else:
        df_comb = pd.DataFrame([row]).set_index('run_time')
    df_comb.reset_index().to_csv(fn, index=False)

# -------------------------
# Ejecución principal
# -------------------------
def main():
    # 1) Actualizar históricos
    for st in station_urls:
        print(f"Histórico {st} ->", update_history(st))

    # 2) Cargar y combinar
    dfs = []
    for st in station_urls:
        df = pd.read_csv(f"{st}.csv", parse_dates=['datetime'])
        df = add_time_feats(df)
        df.set_index('datetime', inplace=True)
        feats = numeric_cols + time_feats
        df = df[feats]
        df.columns = [f"{st}_{c}" for c in feats]
        dfs.append(df)
    merged = reduce(lambda a, b: a.join(b, how='inner'), dfs).sort_index()

    # 3) Normalizar
    stats = pd.read_csv('model/normalization_stats.csv', index_col=0)
    stats.columns = ['mean', 'std']
    merged_norm = merged.copy()
    for col in merged.columns:
        base = col.split('_', 1)[1]
        if base in feature_map:
            key = feature_map[base]
            m, s = stats.at[key, 'mean'], stats.at[key, 'std']
            merged_norm[col] = (merged[col] - m) / s

    # 4) Preparar última ventana
    state = torch.load('model/forecast_model.pt', map_location=DEVICE)
    input_dim = state['encoder.weight_ih_l0'].shape[1]
    hidden_size = state['encoder.weight_ih_l0'].shape[0] // 4
    num_layers = len([k for k in state.keys() if k.startswith('encoder.weight_ih_l')])
    cols = [f"{st}_{feat}" for st in station_urls for feat in numeric_cols + time_feats]
    data = merged_norm[cols].values
    if data.shape[0] < INPUT_WINDOW:
        print(f"No hay suficientes datos ({data.shape[0]}) para ventana {INPUT_WINDOW}h.")
        return
    window = torch.tensor(data[-INPUT_WINDOW:], dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # 5) Predecir próximas 24h
    model = ForecastNet(input_dim, hidden_size, num_layers, OUTPUT_WINDOW, 2).to(DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()
    with torch.no_grad():
        pred = model(window).cpu().numpy().squeeze(0)

    # 6) Compilar resultados
    last_time = merged.index[-1]
    rows = []
    for h in range(1, OUTPUT_WINDOW + 1):
        t_pred = last_time + timedelta(hours=h)
        tp_n, rp_n = pred[h-1]
        m_t, s_t = stats.at['temperature_2m (°C)', 'mean'], stats.at['temperature_2m (°C)', 'std']
        m_r, s_r = stats.at['rain (mm)', 'mean'], stats.at['rain (mm)', 'std']
        temp = tp_n * s_t + m_t
        rain = max(rp_n * s_r + m_r, 0.0)
        rows.append({'time': t_pred, 'horizon_h': h, 'pred_temp': temp, 'pred_rain': rain})
    dfres = pd.DataFrame(rows)

    # 7) Guardar histórico combinado
    os.makedirs('results', exist_ok=True)
    out_fn = 'results/next24h_predictions.csv'
    if os.path.exists(out_fn):
        prev = pd.read_csv(out_fn, parse_dates=['time']).set_index('time')
        combined = pd.concat([prev, dfres.set_index('time')])
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
        idx = pd.date_range(combined.index.min(), combined.index.max(), freq='h')
        combined = combined.reindex(idx)
        combined['pred_temp'] = combined['pred_temp'].interpolate(method='time')
        combined['pred_rain'] = combined['pred_rain'].interpolate(method='time')
    else:
        combined = dfres.set_index('time')
    combined.reset_index().rename(columns={'index':'time'}).to_csv(out_fn, index=False)

    # 8) Graficar últimas 24h
    plt.figure(); plt.plot(dfres.time, dfres.pred_temp, marker='o'); plt.title('Temp +24h'); plt.tight_layout(); plt.savefig('results/next24h_temp.png'); plt.close()
    plt.figure(); plt.plot(dfres.time, dfres.pred_rain, marker='o'); plt.title('Rain +24h'); plt.tight_layout(); plt.savefig('results/next24h_rain.png'); plt.close()

    # 9) Pivot acumulado
    run_time = last_time
    # temperatura
    temp_row = {'run_time': run_time}
    for h in range(1, OUTPUT_WINDOW+1):
        temp_row[str(h)] = dfres.loc[dfres.horizon_h == h, 'pred_temp'].values[0]
    # precipitación
    rain_row = {'run_time': run_time}
    for h in range(1, OUTPUT_WINDOW+1):
        rain_row[str(h)] = dfres.loc[dfres.horizon_h == h, 'pred_rain'].values[0]

    append_and_save(temp_row, 'results/pivot_pred_temp.csv')
    append_and_save(rain_row, 'results/pivot_pred_rain.csv')

    print('Pivot acumulado actualizado en results/')

if __name__ == '__main__':
    main()
