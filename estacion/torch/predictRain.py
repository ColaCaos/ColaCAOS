import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import timedelta
import os

# Configuración de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo utilizado:", device, flush=True)

# Parámetros (deben coincidir con el entrenamiento)
INPUT_WINDOW = 168    # Ventana de entrada: 7 días (168 horas)
OUTPUT_WINDOW = 12    # Ventana de salida: 12 horas

# Lista de características (usando ALL_FEATURES: 6 originales + 4 temporales)
FEATURES = ['temperature_2m (°C)', 'relative_humidity_2m (%)', 'rain (mm)',
            'pressure_msl (hPa)', 'wind_speed_10m (km/h)', 'wind_direction_10m (°)']
TEMPORAL_FEATURES = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
ALL_FEATURES = FEATURES + TEMPORAL_FEATURES

# Parámetros del modelo
HIDDEN_SIZE = 64
NUM_LAYERS = 2

# Variable objetivo: probabilidad de lluvia (valores entre 0 y 1)
OUTPUT_FEATURE = 'rain_probability'

# Modelo (igual que en entrenamiento)
class ForecastNet(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, output_window, dropout=0.0):
        super(ForecastNet, self).__init__()
        self.encoder = nn.LSTM(input_size=num_features, hidden_size=hidden_size, 
                               num_layers=num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        self.decoder = nn.LSTM(input_size=num_features, hidden_size=hidden_size, 
                               num_layers=num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)  # Salida: 1 valor (probabilidad)
        self.proj = nn.Linear(1, num_features)
        self.output_window = output_window

    def forward(self, src):
        # src: [batch, INPUT_WINDOW, num_features]
        _, (hidden, cell) = self.encoder(src)
        # Se usa el último valor de la primera feature (por ejemplo, temperatura) como semilla
        decoder_input = src[:, -1:, 0:1]  # [batch, 1, 1]
        outputs = []
        for t in range(self.output_window):
            decoder_input_proj = self.proj(decoder_input)  # [batch, 1, num_features]
            out, (hidden, cell) = self.decoder(decoder_input_proj, (hidden, cell))
            pred = self.fc(out)  # [batch, 1, 1]
            # Se aplica activación sigmoidal para obtener una salida en [0, 1]
            pred = torch.sigmoid(pred)
            outputs.append(pred)
            decoder_input = pred
        outputs = torch.cat(outputs, dim=1)  # [batch, OUTPUT_WINDOW, 1]
        return outputs

# Función para agregar características temporales (igual que en entrenamiento)
def add_temporal_features(df, time_col='time'):
    df[time_col] = pd.to_datetime(df[time_col])
    df['hour'] = df[time_col].dt.hour
    df['day_of_week'] = df[time_col].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    return df

def main():
    data_path = 'galapagarhoraria25.csv'
    df = pd.read_csv(data_path, skiprows=2, parse_dates=['time'])
    df.sort_values('time', inplace=True)
    
    # Se calcula end_pred usando la columna 'time'
    end_pred = df['time'].max() - pd.Timedelta(hours=OUTPUT_WINDOW)
    
    # Agregar características temporales
    df = add_temporal_features(df, time_col='time')
    
    # Se asume que en el CSV no existe la columna 'rain_probability'. 
    # Se crea a partir de 'rain (mm)' asignando 1 si hay lluvia (> 0) y 0 si no.
    if 'rain_probability' not in df.columns:
        df['rain_probability'] = (df['rain (mm)'] > 0).astype(float)
    
    # Cargar parámetros de normalización
    stats_df = pd.read_csv("model/normalization_statsRain.csv", index_col=0)
    if stats_df.shape[1] != 2 and stats_df.shape[0] == 2:
        stats_df = stats_df.T
    if stats_df.shape[1] != 2:
        raise ValueError(f"El archivo normalization_stats.csv tiene un formato inesperado: {stats_df.shape}")
    if list(stats_df.columns) != ['mean', 'std']:
        stats_df.columns = ['mean', 'std']
    
    # Normalizar los datos de entrada usando ALL_FEATURES
    df_norm = df.copy()
    for col in ALL_FEATURES:
        mean, std = float(stats_df.loc[col, 'mean']), float(stats_df.loc[col, 'std'])
        df_norm[col] = (df_norm[col] - mean) / std
    df_norm.set_index('time', inplace=True)
    
    # Para la evaluación (cotejo) se usa el valor real de lluvia en mm/h (sin normalizar)
    df_raw = df.copy()
    df_raw.set_index('time', inplace=True)
    
    # Cargar el modelo entrenado (se espera que se haya entrenado utilizando ALL_FEATURES y OUTPUT_FEATURE 'rain_probability')
    model = ForecastNet(num_features=len(ALL_FEATURES), hidden_size=HIDDEN_SIZE, 
                        num_layers=NUM_LAYERS, output_window=OUTPUT_WINDOW, dropout=0.2)
    model.load_state_dict(torch.load("model/forecast_modelRain.pt", map_location=device))
    model = model.to(device)
    model.eval()
    
    # Generar predicciones a partir del 8 de enero de 2025 hasta end_pred
    start_pred = pd.Timestamp("2025-01-08 00:00:00")
    forecast_dict = {}
    current_time = start_pred
    while current_time <= end_pred:
        window_start = current_time - pd.Timedelta(hours=INPUT_WINDOW)
        window_data = df_norm.loc[window_start: current_time - pd.Timedelta(hours=1), ALL_FEATURES].values
        if window_data.shape[0] != INPUT_WINDOW:
            current_time += pd.Timedelta(hours=1)
            continue
        x_input = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x_input)  # [1, OUTPUT_WINDOW, 1]
        pred = pred.squeeze(0).cpu().numpy()  # [OUTPUT_WINDOW, 1]
        forecast_dict[current_time] = pred
        current_time += pd.Timedelta(hours=1)
    
    # Comparar las predicciones con los valores reales
    # Se define el valor real como: 1 si la lluvia actual (en mm/h) es > 0, o 0 si es 0
    eval_start = pd.Timestamp("2025-01-11 00:00:00")
    results = []
    current_time = eval_start
    while current_time in df_norm.index:
        try:
            # Valor real (binario) obtenido de 'rain (mm)' sin normalizar
            rain_value = df_raw.loc[current_time, 'rain (mm)']
            real_binary = 1 if rain_value > 0 else 0
            # Se extraen las predicciones de la ventana que comenzó 4, 8 y 12 horas antes
            forecast_4 = forecast_dict[current_time - pd.Timedelta(hours=4)][3][0]
            forecast_8 = forecast_dict[current_time - pd.Timedelta(hours=8)][7][0]
            forecast_12 = forecast_dict[current_time - pd.Timedelta(hours=12)][11][0]
        except KeyError:
            current_time += pd.Timedelta(hours=1)
            continue
        
        results.append({
            'time': current_time,
            'real': real_binary,
            'pred_4': forecast_4,
            'pred_8': forecast_8,
            'pred_12': forecast_12
        })
        current_time += pd.Timedelta(hours=1)
    
    df_res = pd.DataFrame(results)
    
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(12,6))
    plt.plot(df_res['time'], df_res['real'], label='Valor real')
    plt.plot(df_res['time'], df_res['pred_4'], label='Predicción 4h')
    plt.plot(df_res['time'], df_res['pred_8'], label='Predicción 8h')
    plt.plot(df_res['time'], df_res['pred_12'], label='Predicción 12h')
    plt.xlabel('Tiempo')
    plt.ylabel('Probabilidad de lluvia')
    plt.title('Comparación de predicciones para probabilidad de lluvia')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/rain_probability_predictions.png")
    plt.close()
    
    df_res.to_csv("results/rain_probability_predictions.csv", index=False)
    print("Proceso de predicción y cotejo completado. Resultados guardados en la carpeta 'results'.")

if __name__ == '__main__':
    main()
