#!/usr/bin/env python3
"""
Script para generar predicciones en Galapagar usando los últimos 72h de datos de estaciones externas:
  - pontevedra, salamanca, huelva, ciudadreal, valencia, almeria, creus, Santander, Segovia, Guadalajara, Burgos
1) Descargar y actualizar históricos horarios de AEMET para cada estación (excepto Galapagar)
2) Preprocesar datos:
   - completar gaps por interpolación
   - convertir direcciones de viento a grados
   - añadir variables temporales (hora y día de la semana)
   - renombrar a nombres de features del entrenamiento
3) Normalizar según stats de entrenamiento
4) Generar pronóstico hora a hora para las próximas 24h con ForecastNet
5) Guardar resultados en CSV y gráficos
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
# Configuración
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

# Columnas a obtener (solo features usadas en entrenamiento: 6 numéricas + 4 temporales)
numeric_cols = [
    'Temperatura (ºC)',
    'Humedad (%)',
    'Precipitación (mm)',
    'Presión (hPa)',
    'Velocidad del viento (km/h)',
    'Dirección del viento'  # direcciones convertidas a grados
]
# Eliminamos Racha( km/h) y Tendencia(hPa); 'Dirección del viento' se convierte a grados y se trata como numérica
cat_cols = []  # ya convertimos dirección de viento
time_feats = ['hour_sin','hour_cos','dow_sin','dow_cos']

# Mapeo para convertir direcciones de viento a grados
# (usado en update_history)
dir2deg = {
    'Norte': 0, 'Nornoreste': 22.5, 'Noreste': 45, 'Estenoreste': 67.5,
    'Este': 90, 'Estesureste': 112.5, 'Sureste': 135, 'Sursureste': 157.5,
    'Sur': 180, 'Sursuroeste': 202.5, 'Suroeste': 225, 'Oestesuroeste': 247.5,
    'Oeste': 270, 'Oestonoroeste': 292.5, 'Noroeste': 315, 'Nornoroeste': 337.5
}

# Mapeo para stats de entrenamiento (ahora solo numéricas originales) (ahora solo numéricas originales)
feature_map = {
    'Temperatura (ºC)': 'temperature_2m (°C)',
    'Humedad (%)': 'relative_humidity_2m (%)',
    'Precipitación (mm)': 'rain (mm)',
    'Presión (hPa)': 'surface_pressure (hPa)',
    'Velocidad del viento (km/h)': 'wind_speed_10m (km/h)'  
}

# Ventana de entrada (horas) ahora 24 según nuevo modelo
# Ventana de entrada (horas) ahora 24 según nuevo modelo
INPUT_WINDOW = 24
OUTPUT_WINDOW = 24
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------
# Descargar y parsear CSV AEMET
# -------------------------
def fetch_csv(url):
    r = requests.get(url); r.raise_for_status()
    lines = r.text.splitlines()
    for i, l in enumerate(lines):
        if l.startswith('"Fecha y hora oficial"'):
            return pd.read_csv(
                StringIO("\n".join(lines[i:])), sep=',', quotechar='"',
                parse_dates=['Fecha y hora oficial'], dayfirst=True
            )
    raise RuntimeError("Cabecera no encontrada en AEMET.")

# -------------------------
# Actualizar histórico CSV
# -------------------------
def update_history(st):
    fn = f"{st}.csv"
    df_new = fetch_csv(station_urls[st]).set_index('Fecha y hora oficial')
    if os.path.exists(fn):
        df_hist = pd.read_csv(fn, parse_dates=['datetime'], index_col='datetime')
    else:
        df_hist = pd.DataFrame()
    df = pd.concat([df_hist, df_new[numeric_cols + cat_cols]])
    df = df[~df.index.duplicated(keep='last')].sort_index()
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq='h'))
    df[numeric_cols] = df[numeric_cols].interpolate(method='time')
    # convertir viento a grados
    df['Dirección del viento'] = df['Dirección del viento'].map(dir2deg).ffill().bfill()
    df['datetime'] = df.index
    df.to_csv(fn, index=False, float_format='%.1f')
    return fn

# -------------------------
# Añadir variables temporales
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
        # El encoder recibe in_dim features
        self.encoder = nn.LSTM(input_size=in_dim, hidden_size=hid, num_layers=nl,
                                batch_first=True, dropout=drop)
        # El decoder debe recibir también in_dim features para cargar el checkpoint correctamente
        self.decoder = nn.LSTM(input_size=in_dim, hidden_size=hid, num_layers=nl,
                                batch_first=True, dropout=drop)
        self.fc   = nn.Linear(hid, od)
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
        return torch.cat(outs, dim=1)

# -------------------------
# Pipeline principal
# -------------------------
def main():
    # 1) actualizar históricos
    for st in station_urls:
        print(f"Histórico {st} ->", update_history(st))

    # 2) cargar y combinar datos
    dfs = []
    for st in station_urls:
        df = pd.read_csv(f"{st}.csv", parse_dates=['datetime'])
        df = add_time_feats(df)
        df.set_index('datetime', inplace=True)
        feats = numeric_cols + cat_cols + time_feats
        df = df[feats]
        df.columns = [f"{st}_{c}" for c in feats]
        dfs.append(df)
    merged = reduce(lambda a, b: a.join(b, how='inner'), dfs).sort_index()

    # 3) normalizar
    stats = pd.read_csv('model/normalization_stats.csv', index_col=0)
    stats.columns = ['mean', 'std']
    merged_norm = merged.copy()
    for col in merged.columns:
        base = col.split('_', 1)[1]
        if base in feature_map:
            key = feature_map[base]
            if key in stats.index:
                m, s = stats.at[key, 'mean'], stats.at[key, 'std']
                merged_norm[col] = (merged[col] - m) / s

    # 4) preparar datos de entrada con orden consistente de features
    state = torch.load('model/forecast_model.pt', map_location=DEVICE)
    input_dim = state['encoder.weight_ih_l0'].shape[1]
    hidden_size = state['encoder.weight_ih_l0'].shape[0] // 4
    num_layers = len([k for k in state.keys() if k.startswith('encoder.weight_ih_l')])
    base_feats = numeric_cols + cat_cols + time_feats
    input_cols = []
    for st in station_urls:
        for feat in base_feats:
            col = f"{st}_{feat}"
            if col not in merged_norm.columns:
                raise KeyError(f"Columna faltante en merged_norm: {col}")
            input_cols.append(col)
    if len(input_cols) != input_dim:
        raise ValueError(f"Número de columnas de entrada {len(input_cols)} no coincide con input_dim {input_dim}")
    data = merged_norm[input_cols].values
    if data.shape[0] < INPUT_WINDOW:
        print(f"No hay suficientes datos ({data.shape[0]}) para ventana {INPUT_WINDOW}h.")
        return
    last_win = torch.tensor(data[-INPUT_WINDOW:], dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # 5) predecir próximas 24h
    model = ForecastNet(input_dim, hidden_size, num_layers, OUTPUT_WINDOW, 2).to(DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()
    with torch.no_grad():
        pred = model(last_win).cpu().numpy().squeeze(0)

    # 6) compilar pronóstico hora a hora
    last_time = merged.index[-1]
    rows = []
    for h in range(1, OUTPUT_WINDOW+1):
        t_pred = last_time + timedelta(hours=h)
        tp_n, rp_n = pred[h-1]
        m_t, s_t = stats.at['temperature_2m (°C)', 'mean'], stats.at['temperature_2m (°C)', 'std']
        m_r, s_r = stats.at['rain (mm)', 'mean'], stats.at['rain (mm)', 'std']
        temp = tp_n * s_t + m_t
        rain = max(rp_n * s_r + m_r, 0.0)
        rows.append({'time': t_pred, 'horizon_h': h, 'pred_temp': temp, 'pred_rain': rain})
        dfres = pd.DataFrame(rows)
    os.makedirs('results', exist_ok=True)
    out_fn = 'results/next24h_predictions.csv'
    # Cargar anteriores predicciones y unir sin duplicados
    if os.path.exists(out_fn):
        prev = pd.read_csv(out_fn, parse_dates=['time']).set_index('time')
    else:
        prev = None
    new = dfres.set_index('time')
    if prev is not None:
        combined = pd.concat([prev, new])
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
        # rellenar huecos por interpolación horaria
        idx = pd.date_range(combined.index.min(), combined.index.max(), freq='h')
        combined = combined.reindex(idx)
        combined['pred_temp'] = combined['pred_temp'].interpolate(method='time')
        combined['pred_rain'] = combined['pred_rain'].interpolate(method='time')
    else:
        combined = new
    # Guardar predicciones actualizadas
    combined = combined.reset_index().rename(columns={'index':'time'})
    combined.to_csv(out_fn, index=False)

        # 7) graficar últimas 24h de predicción
    plt.figure(); plt.plot(dfres.time, dfres.pred_temp, marker='o'); plt.title('Temp +24h'); plt.tight_layout(); plt.savefig('results/next24h_temp.png'); plt.close()
    plt.figure(); plt.plot(dfres.time, dfres.pred_rain, marker='o'); plt.title('Rain +24h'); plt.tight_layout(); plt.savefig('results/next24h_rain.png'); plt.close()

    print('Pronóstico hora a hora generado en results/next24h_predictions.csv')

    # 8) Generar tabla pivote de pronósticos históricos
    # Cargar CSV combinado
    df_all = pd.read_csv(out_fn, parse_dates=['time'])
    # Pivot para temperatura
    pivot_temp = df_all.pivot(index='time', columns='horizon_h', values='pred_temp')
    pivot_temp = pivot_temp.sort_index()
    # Asegurar índice continuo hora a hora
    idx = pd.date_range(pivot_temp.index.min(), pivot_temp.index.max(), freq='h')
    pivot_temp = pivot_temp.reindex(idx)
    # Rellenar gaps con NaN o interpolación si se desea

    # Pivot para precipitación
    pivot_rain = df_all.pivot(index='time', columns='horizon_h', values='pred_rain')
    pivot_rain = pivot_rain.sort_index()
    pivot_rain = pivot_rain.reindex(idx)

    # Guardar tablas pivote
    pivot_temp.to_csv('results/pivot_pred_temp.csv', index_label='time')
    pivot_rain.to_csv('results/pivot_pred_rain.csv', index_label='time')

    print('Tablas pivote guardadas en results/pivot_pred_temp.csv y pivot_pred_rain.csv')

if __name__ == '__main__':
    main()
    main()
