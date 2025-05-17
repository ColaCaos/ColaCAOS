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
    'santander':   'https://www.aemet.es/es/eltiempo/observacion/ultimosdatos_1111X_datos-horarios.csv?k=can&l=1111X&datos=det&w=0&f=temperatura&x=h24',
    'segovia':     'https://www.aemet.es/es/eltiempo/observacion/ultimosdatos_2465_datos-horarios.csv?k=cle&l=2465&datos=det&w=0&f=temperatura&x=h24',
    'guadalajara': 'https://www.aemet.es/es/eltiempo/observacion/ultimosdatos_3168D_datos-horarios.csv?k=clm&l=3168D&datos=det&w=0&f=temperatura&x=h24',
    'burgos':      'https://www.aemet.es/es/eltiempo/observacion/ultimosdatos_2331_datos-horarios.csv?k=cle&l=2331&datos=det&w=0&f=temperatura&x=h24',
    'galapagar':   'https://www.aemet.es/es/eltiempo/observacion/ultimosdatos_3272M_datos-horarios.csv?k=mad&l=3272M&datos=det&w=0&f=temperatura&x='
}

numeric_cols = [
    'Temperatura (ºC)', 'Humedad (%)', 'Precipitación (mm)',
    'Presión (hPa)', 'Velocidad del viento (km/h)', 'Dirección del viento'
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
    'Velocidad del viento (km/h)': 'wind_speed_10m (km/h)',
    'Dirección del viento': 'wind_direction_10m (°)'
}
INPUT_WINDOW = 24
OUTPUT_WINDOW = 24
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fetch_csv(url):
    r = requests.get(url); r.raise_for_status()
    for i, l in enumerate(r.text.splitlines()):
        if l.startswith('"Fecha y hora oficial"'):
            return pd.read_csv(StringIO("\n".join(r.text.splitlines()[i:])),
                               sep=',', quotechar='"',
                               parse_dates=['Fecha y hora oficial'], dayfirst=True)
    raise RuntimeError("Cabecera no encontrada en AEMET.")

def update_history(st):
    fn = f"{st}.csv"
    df_new = fetch_csv(station_urls[st]).set_index('Fecha y hora oficial')
    df_new = df_new[numeric_cols]
    df_new['Dirección del viento'] = df_new['Dirección del viento'].map(dir2deg)
    df_new['datetime'] = df_new.index
    if os.path.exists(fn):
        df_old = pd.read_csv(fn, parse_dates=['datetime']).set_index('datetime')
        df_comb = pd.concat([df_old, df_new], axis=0)
        # Eliminar duplicados en índice combinado
        df_comb = df_comb[~df_comb.index.duplicated(keep='last')]
    else:
        df_comb = df_new
    df_comb = df_comb.reindex(pd.date_range(df_comb.index.min(), df_comb.index.max(), freq='h'))
    df_comb[numeric_cols] = df_comb[numeric_cols].interpolate(method='time')
    df_comb['Dirección del viento'] = df_comb['Dirección del viento'].ffill().bfill()
    df_final = df_comb.tail(INPUT_WINDOW)
    df_final['datetime'] = df_final.index
    df_final.to_csv(fn, index=False, float_format='%.1f')
    return fn

def add_time_feats(df):
    t = df['datetime']
    df['hour_sin'] = np.sin(2*np.pi*t.dt.hour/24)
    df['hour_cos'] = np.cos(2*np.pi*t.dt.hour/24)
    df['dow_sin'] = np.sin(2*np.pi*t.dt.dayofweek/7)
    df['dow_cos'] = np.cos(2*np.pi*t.dt.dayofweek/7)
    return df

class ForecastNet(nn.Module):
    def __init__(self, in_dim, hid, nl, steps, od, drop=0.0):
        super().__init__()
        self.encoder = nn.LSTM(in_dim, hid, nl, batch_first=True, dropout=drop if nl>1 else 0)
        self.decoder = nn.LSTM(in_dim, hid, nl, batch_first=True, dropout=drop if nl>1 else 0)
        self.fc = nn.Linear(hid, od)
        self.proj = nn.Linear(od, in_dim)
        self.steps = steps

def main():
    for st in station_urls:
        print(f"Actualizado histórico {st} ->", update_history(st))
    df_g = pd.read_csv('galapagar.csv', parse_dates=['datetime'])
    df_g = add_time_feats(df_g).set_index('datetime')
    time_index = df_g.index
    dfs = []
    for st in station_urls:
        df = pd.read_csv(f"{st}.csv", parse_dates=['datetime'])
        df = add_time_feats(df).set_index('datetime')
        df = df.reindex(time_index)
        df[numeric_cols] = df[numeric_cols].interpolate(method='time')
        df['Dirección del viento'] = df['Dirección del viento'].ffill().bfill()
        feats = numeric_cols + time_feats
        df = df[feats]
        df.columns = [f"{st}_{c}" for c in feats]
        dfs.append(df)
    merged = pd.concat(dfs, axis=1).sort_index()
    stats = pd.read_csv('model/normalization_stats.csv', index_col=0)
    stats.columns = ['mean','std']
    merged_norm = merged.copy()
    input_cols = []
    for col in merged.columns:
        station, base = col.split('_',1)
        eng = feature_map.get(base, base)
        key = f"{station.lower()}_{eng}" if station!='galapagar' else eng
        m, s = stats.at[key,'mean'], stats.at[key,'std']
        merged_norm[col] = (merged[col]-m)/s
        input_cols.append(col)
    state = torch.load('model/forecast_model.pt', map_location=DEVICE)
    input_dim = state['encoder.weight_ih_l0'].shape[1]
    hidden = state['encoder.weight_ih_l0'].shape[0]//4
    layers = len([k for k in state if k.startswith('encoder.weight_ih_l')])
    if len(input_cols)!=input_dim:
        raise ValueError(f"Dim entrada {len(input_cols)} != {input_dim}")
    data = merged_norm[input_cols].values
    last_win = torch.tensor(data[-INPUT_WINDOW:],dtype=torch.float32).unsqueeze(0).to(DEVICE)
    model = ForecastNet(input_dim, hidden, layers, OUTPUT_WINDOW, 2)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    idx_t = input_cols.index('galapagar_Temperatura (ºC)')
    idx_r = input_cols.index('galapagar_Precipitación (mm)')
    with torch.no_grad():
        _,(h,c)=model.encoder(last_win)
        inp = last_win[:,-1:,[idx_t,idx_r]]
        outs=[]
        for _ in range(OUTPUT_WINDOW):
            d=model.proj(inp)
            out,(h,c)=model.decoder(d,(h,c))
            p=model.fc(out)
            outs.append(p)
            inp=p
        pred = torch.cat(outs,1).cpu().numpy().squeeze(0)
    rows=[]
    last_time = merged.index[-1]
    temps=[]
    rains=[]
    for h in range(OUTPUT_WINDOW):
        tp, rp = pred[h]
        m_t, s_t = stats.at['temperature_2m (°C)','mean'], stats.at['temperature_2m (°C)','std']
        m_r, s_r = stats.at['rain (mm)','mean'], stats.at['rain (mm)','std']
        val_t = tp*s_t + m_t
        val_r = max(rp*s_r + m_r, 0.0)
        if val_r < 0.1:
            val_r = 0.0
        rows.append({'time': last_time + timedelta(hours=h+1), 'horizon_h': h+1,
                     'pred_temp': val_t, 'pred_rain': val_r})
        temps.append(val_t)
        rains.append(val_r)
    dfres = pd.DataFrame(rows)
    os.makedirs('results', exist_ok=True)
    dfres.to_csv('results/next24h_predictions.csv', index=False)
    # Serie de tiempo
    plt.figure(); plt.plot(dfres.time, dfres.pred_temp, marker='o'); plt.title('Temperatura +24h'); plt.tight_layout(); plt.savefig('results/next24h_temp.png'); plt.close()
    plt.figure(); plt.plot(dfres.time, dfres.pred_rain, marker='o'); plt.title('Precipitación +24h'); plt.tight_layout(); plt.savefig('results/next24h_rain.png'); plt.close()
    # 7) Guardar filas anchas sin pivot
    temp_keys = [f'pred_temp{i+1}h' for i in range(OUTPUT_WINDOW)]
    rain_keys = [f'pred_rain{i+1}h' for i in range(OUTPUT_WINDOW)]
    # Crear DataFrame de una sola fila
    temp_row = pd.DataFrame([[last_time] + temps], columns=['time'] + temp_keys)
    rain_row = pd.DataFrame([[last_time] + rains], columns=['time'] + rain_keys)
    # Append a CSV existente o crear nuevo
    tfile = 'results/temperature_24h_wide.csv'
    if os.path.exists(tfile):
        df_temp_existing = pd.read_csv(tfile, parse_dates=['time'])
        df_temp_existing = pd.concat([df_temp_existing, temp_row], ignore_index=True)
        df_temp_existing.to_csv(tfile, index=False)
    else:
        temp_row.to_csv(tfile, index=False)
    rfile = 'results/rain_24h_wide.csv'
    if os.path.exists(rfile):
        df_rain_existing = pd.read_csv(rfile, parse_dates=['time'])
        df_rain_existing = pd.concat([df_rain_existing, rain_row], ignore_index=True)
        df_rain_existing.to_csv(rfile, index=False)
    else:
        rain_row.to_csv(rfile, index=False)
    print('Pronóstico +24h generado y ficheros anchos actualizados.')

if __name__ == '__main__':
    main()
