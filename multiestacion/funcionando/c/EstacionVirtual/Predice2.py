#!/usr/bin/env python3
"""
Script corregido para generar predicciones +24h en Galapagar (usando datos de Torrelodones) replicando la lógica de predictMultiAMPmetrics.py.
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
# Configuración de estaciones (Torrelodones como Galapagar)
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
    # Galapagar -> Torrelodones (código 3272M)
    'galapagar':   'https://www.aemet.es/es/eltiempo/observacion/ultimosdatos_3272M_datos-horarios.csv?k=mad&l=3272M&datos=det&w=0&f=temperatura&x='
}

# Columnas de interés y mapeos
numeric_cols = [
    'Temperatura (ºC)', 'Humedad (%)', 'Precipitación (mm)',
    'Presión (hPa)', 'Velocidad del viento (km/h)', 'Dirección del viento'
]
cat_cols = []  # la dirección ya se convierte a grados
time_feats = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']

dir2deg = {
    'Norte': 0, 'Nornoreste': 22.5, 'Noreste': 45, 'Estenoreste': 67.5,
    'Este': 90, 'Estesureste': 112.5, 'Sureste': 135, 'Sursureste': 157.5,
    'Sur': 180, 'Sursuroeste': 202.5, 'Suroeste': 225, 'Oestesuroeste': 247.5,
    'Oeste': 270, 'Oestonoroeste': 292.5, 'Noroeste': 315, 'Nornoroeste': 337.5
}

# Mapa de nombres de características a índices de stats
feature_map = {
    'Temperatura (ºC)': 'temperature_2m (°C)',
    'Humedad (%)': 'relative_humidity_2m (%)',
    'Precipitación (mm)': 'rain (mm)',
    'Presión (hPa)': 'surface_pressure (hPa)',
    'Velocidad del viento (km/h)': 'wind_speed_10m (km/h)',
    'Dirección del viento': 'wind_direction_10m (°)'
}

# Ventanas y dispositivo
INPUT_WINDOW = 24
OUTPUT_WINDOW = 24
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------
# Funciones auxiliares
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


def update_history(st):
    """
    Descarga los datos de AEMET para la estación st, 
    mantiene sólo las últimas INPUT_WINDOW horas, 
    interpola huecos, codifica la dirección del viento
    y reescribe el fichero 'st.csv'.
    """
    fn = f"{st}.csv"
    # 1) Descargar y parsear df_new
    df_new = fetch_csv(station_urls[st]).set_index('Fecha y hora oficial')
    df_new = df_new[numeric_cols + cat_cols]
    # 2) Convertir direcciones y renombrar datetime
    df_new['Dirección del viento'] = df_new['Dirección del viento'].map(dir2deg)
    df_new['datetime'] = df_new.index
    # 3) Si hay histórico previo, cargarlo y añadir nuevos datos
    if os.path.exists(fn):
        df_old = pd.read_csv(fn, parse_dates=['datetime']).set_index('datetime')
        df_comb = pd.concat([df_old, df_new], axis=0)
        # Eliminar duplicados manteniendo el más reciente
        df_comb = df_comb[~df_comb.index.duplicated(keep='last')]
    else:
        df_comb = df_new
    # 4) Reindexar para frecuencia horaria continua
    df_comb = df_comb.reindex(pd.date_range(df_comb.index.min(),
                                            df_comb.index.max(),
                                            freq='h'))
    # 5) Interpolar numéricos y rellenar dirección
    df_comb[numeric_cols] = df_comb[numeric_cols].interpolate(method='time')
    df_comb['Dirección del viento'] = df_comb['Dirección del viento'].ffill().bfill()
    # 6) Mantener sólo las últimas INPUT_WINDOW horas
    df_final = df_comb.tail(INPUT_WINDOW)
    # 7) Preparar columna datetime y escribir
    df_final = df_final.copy()
    df_final['datetime'] = df_final.index
    df_final.to_csv(fn, index=False, float_format='%.1f')
    return fn


def add_time_feats(df):
    t = df['datetime']
    df['hour_sin'] = np.sin(2*np.pi*t.dt.hour/24)
    df['hour_cos'] = np.cos(2*np.pi*t.dt.hour/24)
    df['dow_sin']  = np.sin(2*np.pi*t.dt.dayofweek/7)
    df['dow_cos']  = np.cos(2*np.pi*t.dt.dayofweek/7)
    return df

# -------------------------
# Modelo ForecastNet
# -------------------------
class ForecastNet(nn.Module):
    def __init__(self, in_dim, hid, nl, steps, od, drop=0.0):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=in_dim, hidden_size=hid, num_layers=nl,
            batch_first=True, dropout=drop if nl>1 else 0)
        self.decoder = nn.LSTM(
            input_size=in_dim, hidden_size=hid, num_layers=nl,
            batch_first=True, dropout=drop if nl>1 else 0)
        self.fc   = nn.Linear(hid, od)
        self.proj = nn.Linear(od, in_dim)
        self.steps = steps

# -------------------------
# Pipeline principal
# -------------------------

def main():
    # 1) Actualizar CSVs
    for st in station_urls:
        print(f"Actualizado histórico {st} ->", update_history(st))

    # 2) Cargar y unir usando índice de Galapagar
    # Leer Galapagar para definir el índice horario común
    df_g = pd.read_csv('galapagar.csv', parse_dates=['datetime'])
    df_g = add_time_feats(df_g).set_index('datetime')
    time_index = df_g.index

    dfs = []
    for st in station_urls:
        df = pd.read_csv(f"{st}.csv", parse_dates=['datetime'])
        df = add_time_feats(df).set_index('datetime')
        # Reindexar al índice común de Galapagar
        df = df.reindex(time_index)
        # Interpolar valores numéricos y rellenar dirección
        df[numeric_cols] = df[numeric_cols].interpolate(method='time')
        df['Dirección del viento'] = df['Dirección del viento'].ffill().bfill()
        # Seleccionar columnas y renombrar con prefijo de estación
        feats = numeric_cols + cat_cols + time_feats
        df = df[feats]
        df.columns = [f"{st}_{c}" for c in feats]
        dfs.append(df)
    merged = pd.concat(dfs, axis=1).sort_index()

# 3) Normalizar todas las features (numéricas y temporales)
    stats = pd.read_csv('model/normalization_stats.csv', index_col=0)
    if list(stats.columns) != ['mean','std']:
        stats.columns = ['mean','std']
    merged_norm = merged.copy()
    input_cols = []
    # Normalizar con estadísticas por estación y característica
    for col in merged.columns:
        station, base = col.split('_', 1)
        # Mapear base en español a clave en stats
        if base in feature_map:
            eng_base = feature_map[base]
        elif base in time_feats:
            eng_base = base
        else:
            continue
        # Construir clave de stats usando nombre de estación en minúsculas (salvo galapagar)
        if station.lower() != 'galapagar':
            key = f"{station.lower()}_{eng_base}"
        else:
            key = eng_base
        if key not in stats.index:
            raise KeyError(f"Clave de normalización no encontrada: {key}")
        # Extraer media y std
        m, s = stats.at[key, 'mean'], stats.at[key, 'std']
        # Normalizar
        merged_norm[col] = (merged[col] - m) / s
        input_cols.append(col)

# 4) Preparar ventana de entrada
    state = torch.load('model/forecast_model.pt', map_location=DEVICE)
    input_dim = state['encoder.weight_ih_l0'].shape[1]
    hidden_size = state['encoder.weight_ih_l0'].shape[0] // 4
    num_layers = len([k for k in state if k.startswith('encoder.weight_ih_l')])

    if len(input_cols) != input_dim:
        raise ValueError(f"Dim entrada {len(input_cols)} != esperado {input_dim}")

    data = merged_norm[input_cols].values
    if data.shape[0] < INPUT_WINDOW:
        raise RuntimeError("No hay suficientes datos para ventana de 24h.")
    last_win = torch.tensor(data[-INPUT_WINDOW:], dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # 5) Cargar modelo y generar +24h con iniciales de Galapagar
    model = ForecastNet(input_dim, hidden_size, num_layers, OUTPUT_WINDOW, 2)
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE)
    model.eval()

    # índices de temperatura y lluvia de Galapagar
    idx_t = input_cols.index('galapagar_Temperatura (ºC)')
    idx_r = input_cols.index('galapagar_Precipitación (mm)')
    with torch.no_grad():
        _, (h, c) = model.encoder(last_win)
        inp = last_win[:, -1:, [idx_t, idx_r]]  # [1,1,2]
        outs = []
        for _ in range(OUTPUT_WINDOW):
            d = model.proj(inp)
            out, (h, c) = model.decoder(d, (h, c))
            p = model.fc(out)  # [1,1,2]
            outs.append(p)
            inp = p
        pred = torch.cat(outs, dim=1).cpu().numpy().squeeze(0)

    # 6) Desnormalizar y guardar resultados
    rows = []
    last_time = merged.index[-1]
    for hstep in range(1, OUTPUT_WINDOW+1):
        tm = last_time + timedelta(hours=hstep)
        tp_n, rp_n = pred[hstep-1]
        m_t, s_t = stats.at['temperature_2m (°C)','mean'], stats.at['temperature_2m (°C)','std']
        m_r, s_r = stats.at['rain (mm)','mean'], stats.at['rain (mm)','std']
        # Desnormalización
        temp_val = tp_n * s_t + m_t
        rain_val = rp_n * s_r + m_r
        # Aplicar umbral: nada si < 0.1 mm/h
        if rain_val < 0.1:
            rain_val = 0.0
        rows.append({
            'time': tm,
            'horizon_h': hstep,
            'pred_temp': temp_val,
            'pred_rain': rain_val
        })
    dfres = pd.DataFrame(rows)
    os.makedirs('results', exist_ok=True)
    dfres.to_csv('results/next24h_predictions.csv', index=False)
    plt.figure(); plt.plot(dfres.time, dfres.pred_temp, marker='o'); plt.title('Temperatura +24h'); plt.tight_layout(); plt.savefig('results/next24h_temp.png'); plt.close()
    plt.figure(); plt.plot(dfres.time, dfres.pred_rain, marker='o'); plt.title('Precipitación +24h'); plt.tight_layout(); plt.savefig('results/next24h_rain.png'); plt.close()
    print('Pronóstico +24h generado: results/next24h_predictions.csv')

if __name__ == '__main__':
    main()
