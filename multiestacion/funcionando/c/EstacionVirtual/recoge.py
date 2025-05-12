#!/usr/bin/env python3
"""
Script para descargar los últimos datos horarios de AEMET para varias localidades,
actualizar históricos en ficheros CSV individuales, evitar duplicados y rellenar huecos
con interpolación (numéricos) y propagación (categóricos) para tener serie continua.

Ejecútalo cada 24h (p. ej. en crontab entre las 21:00 y 24:00).
"""
import os
import pandas as pd
import requests
from io import StringIO

# Configuración de estaciones: nombre legible y URL de AEMET
stations = {
    'Torrelodones': 'https://www.aemet.es/es/eltiempo/observacion/ultimosdatos_3272M_datos-horarios.csv?k=mad&l=3272M&datos=det&w=0&f=temperatura&x=',
    'Pontevedra': 'https://www.aemet.es/es/eltiempo/observacion/ultimosdatos_1484C_datos-horarios.csv?k=gal&l=1484C&datos=det&w=0&f=temperatura&x=',
    'Salamanca': 'https://www.aemet.es/es/eltiempo/observacion/ultimosdatos_2870_datos-horarios.csv?k=cle&l=2870&datos=det&w=0&f=temperatura&x=',
    'Huelva': 'https://www.aemet.es/es/eltiempo/observacion/ultimosdatos_4642E_datos-horarios.csv?k=and&l=4642E&datos=det&w=0&f=temperatura&x=',
    'CiudadReal': 'https://www.aemet.es/es/eltiempo/observacion/ultimosdatos_4121_datos-horarios.csv?k=clm&l=4121&datos=det&w=0&f=temperatura&x=',
    'Valencia': 'https://www.aemet.es/es/eltiempo/observacion/ultimosdatos_8414A_datos-horarios.csv?k=val&l=8414A&datos=det&w=0&f=temperatura&x=',
    'Almeria': 'https://www.aemet.es/es/eltiempo/observacion/ultimosdatos_6325O_datos-horarios.csv?k=and&l=6325O&datos=det&w=0&f=temperatura&x=',
    'CalaCreus': 'https://www.aemet.es/es/eltiempo/observacion/ultimosdatos_0433D_datos-horarios.csv?k=cat&l=0433D&datos=det&w=0&f=temperatura&x=h24',
    'Santander': 'https://www.aemet.es/es/eltiempo/observacion/ultimosdatos_1111X_datos-horarios.csv?k=can&l=1111X&datos=det&w=0&f=temperatura&x=h24',
    'Segovia': 'https://www.aemet.es/es/eltiempo/observacion/ultimosdatos_2482B_datos-horarios.csv?k=cle&l=2482B&datos=det&w=0&f=temperatura&x=h24',
    'Guadalajara': 'https://www.aemet.es/es/eltiempo/observacion/ultimosdatos_3168D_datos-horarios.csv?k=clm&l=3168D&datos=det&w=0&f=temperatura&x=h24',
    'Burgos': 'https://www.aemet.es/es/eltiempo/observacion/ultimosdatos_2331_datos-horarios.csv?k=cle&l=2331&datos=det&w=0&f=temperatura&x=h24'
}

# Columnas originales del CSV de AEMET (categóricas vs. numéricas)
numeric_cols = [
    'Temperatura (ºC)', 'Velocidad del viento (km/h)', 'Racha (km/h)',
    'Precipitación (mm)', 'Presión (hPa)', 'Tendencia (hPa)', 'Humedad (%)'
]
cat_cols = ['Dirección del viento', 'Dirección de racha']

def fetch_station(name, url):
    """Descarga y parsea el CSV horario de AEMET para una estación."""
    resp = requests.get(url)
    resp.raise_for_status()
    text = resp.text
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.startswith('"Fecha y hora oficial"'):
            header_idx = idx
            break
    else:
        raise ValueError(f"Cabecera no encontrada en datos de {name}")
    csv_data = "\n".join(lines[header_idx:])
    df = pd.read_csv(
        StringIO(csv_data),
        sep=',',
        quotechar='"',
        decimal='.',
        parse_dates=['Fecha y hora oficial'],
        dayfirst=True
    )
    df = df.rename(columns={'Fecha y hora oficial': 'datetime'})
    df['datetime'] = df['datetime'].dt.round('H')
    df = df.set_index('datetime')
    for col in numeric_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def load_history(path):
    if os.path.exists(path):
        hist = pd.read_csv(
            path,
            sep=',',
            quotechar='"',
            decimal='.',
            parse_dates=['Fecha y hora oficial'],
            dayfirst=True
        )
        hist = hist.rename(columns={'Fecha y hora oficial': 'datetime'})
        hist = hist.set_index('datetime')
        return hist
    else:
        return pd.DataFrame()


def save_history(df, path):
    df_out = df.copy()
    df_out['Fecha y hora oficial'] = df_out.index.strftime('%d/%m/%Y %H:%M')
    cols = ['Fecha y hora oficial'] + [c for c in df.columns if c in numeric_cols + cat_cols]
    df_out = df_out[cols]
    df_out.to_csv(path, index=False, float_format='%.1f')


def update_station(name, url):
    print(f"Procesando {name}...")
    hist_path = f"{name.replace(' ', '_')}.csv"
    try:
        df_new = fetch_station(name, url)
        df_hist = load_history(hist_path)
        df = pd.concat([df_hist, df_new])
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()
        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H')
        df = df.reindex(full_idx)
        df[numeric_cols] = df[numeric_cols].interpolate(method='time')
        df[cat_cols] = df[cat_cols].fillna(method='ffill').fillna(method='bfill')
        save_history(df, hist_path)
        print(f"Histórico actualizado: {hist_path} ({len(df)} registros)")
    except Exception as e:
        print(f"Error al actualizar {name}: {e}")


def main():
    for name, url in stations.items():
        update_station(name, url)


if __name__ == '__main__':
    main()
