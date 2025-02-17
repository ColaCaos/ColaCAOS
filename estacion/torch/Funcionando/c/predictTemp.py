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
INPUT_WINDOW = 168    # Ventana de entrada: 7 días
OUTPUT_WINDOW = 12    # Ventana de salida: 12 horas
FEATURES = ['temperature_2m (°C)', 'relative_humidity_2m (%)', 'rain (mm)',
            'pressure_msl (hPa)', 'wind_speed_10m (km/h)', 'wind_direction_10m (°)']
HIDDEN_SIZE = 64
NUM_LAYERS = 2

# Variable objetivo
OUTPUT_FEATURE = 'temperature_2m (°C)'

# Modelo: mismo que en entrenamiento, adaptado para salida 1
class ForecastNet(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, output_window, dropout=0.0):
        super(ForecastNet, self).__init__()
        self.encoder = nn.LSTM(input_size=num_features, hidden_size=hidden_size, 
                               num_layers=num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        self.decoder = nn.LSTM(input_size=num_features, hidden_size=hidden_size, 
                               num_layers=num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)  # salida: 1 valor (temperatura)
        self.output_window = output_window
        self.proj = nn.Linear(1, num_features)
        
    def forward(self, src):
        _, (hidden, cell) = self.encoder(src)
        # Usar el último valor de temperatura del input (columna 0) como semilla
        decoder_input = src[:, -1:, 0:1]  # [batch, 1, 1]
        outputs = []
        for t in range(self.output_window):
            decoder_input_proj = self.proj(decoder_input)  # [batch, 1, num_features]
            out, (hidden, cell) = self.decoder(decoder_input_proj, (hidden, cell))
            pred = self.fc(out)  # [batch, 1, 1]
            outputs.append(pred)
            decoder_input = pred
        outputs = torch.cat(outputs, dim=1)  # [batch, OUTPUT_WINDOW, 1]
        return outputs

# Función para desnormalizar (solo temperatura)
def denormalize(series, mean, std):
    return series * std + mean

def main():
    data_path = 'galapagarhoraria25.csv'
    df = pd.read_csv(data_path, skiprows=2, parse_dates=['time'])
    df.sort_values('time', inplace=True)
    
    # Cargar parámetros de normalización
    stats_df = pd.read_csv("model/normalization_stats.csv", index_col=0)
    if stats_df.shape[1] != 2 and stats_df.shape[0] == 2:
        stats_df = stats_df.T
    if stats_df.shape[1] != 2:
        raise ValueError(f"El archivo normalization_stats.csv tiene un formato inesperado: {stats_df.shape}")
    if list(stats_df.columns) != ['mean', 'std']:
        stats_df.columns = ['mean', 'std']
    
    # Extraer estadísticas solo para la temperatura
    try:
        temp_mean = float(stats_df.loc[OUTPUT_FEATURE, 'mean'])
        temp_std = float(stats_df.loc[OUTPUT_FEATURE, 'std'])
    except KeyError:
        raise KeyError(f"La variable {OUTPUT_FEATURE} no se encontró en normalization_stats.csv")
    
    # Normalizar los datos de 2025 (todas las features para el input)
    df_norm = df.copy()
    for col in FEATURES:
        mean, std = float(stats_df.loc[col, 'mean']), float(stats_df.loc[col, 'std'])
        df_norm[col] = (df_norm[col] - mean) / std
    
    # Cargar el modelo entrenado
    model = ForecastNet(num_features=len(FEATURES), hidden_size=HIDDEN_SIZE, 
                        num_layers=NUM_LAYERS, output_window=OUTPUT_WINDOW, dropout=0.2)
    model.load_state_dict(torch.load("model/forecast_model.pt", map_location=device))
    model = model.to(device)
    model.eval()
    
    # Generar predicciones a partir del 8 de enero de 2025
    start_pred = pd.Timestamp("2025-01-08 00:00:00")
    end_pred = df_norm['time'].max() - pd.Timedelta(hours=OUTPUT_WINDOW)
    forecast_dict = {}
    df_norm.set_index('time', inplace=True)
    
    current_time = start_pred
    while current_time <= end_pred:
        window_start = current_time - pd.Timedelta(hours=INPUT_WINDOW)
        window_data = df_norm.loc[window_start: current_time - pd.Timedelta(hours=1), FEATURES].values
        if window_data.shape[0] != INPUT_WINDOW:
            current_time += pd.Timedelta(hours=1)
            continue
        x_input = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x_input)  # [1, OUTPUT_WINDOW, 1]
        pred = pred.squeeze(0).cpu().numpy()  # [OUTPUT_WINDOW, 1]
        forecast_dict[current_time] = pred
        current_time += pd.Timedelta(hours=1)
    
    # Cotejo de predicciones con valores reales (evaluación a partir del 11 de enero de 2025)
    eval_start = pd.Timestamp("2025-01-11 00:00:00")
    results = []
    current_time = eval_start
    while current_time in df_norm.index:
        try:
            real = df_norm.loc[current_time, OUTPUT_FEATURE]
            forecast_4 = forecast_dict[current_time - pd.Timedelta(hours=4)][3][0]
            forecast_8 = forecast_dict[current_time - pd.Timedelta(hours=8)][7][0]
            forecast_12 = forecast_dict[current_time - pd.Timedelta(hours=12)][11][0]
        except KeyError:
            current_time += pd.Timedelta(hours=1)
            continue
        
        results.append({
            'time': current_time,
            'real': real,
            'pred_4': forecast_4,
            'pred_8': forecast_8,
            'pred_12': forecast_12
        })
        current_time += pd.Timedelta(hours=1)
    
    df_res = pd.DataFrame(results)
    # Desnormalizar (usando las estadísticas de temperatura)
    df_res['real'] = denormalize(df_res['real'], temp_mean, temp_std)
    df_res['pred_4'] = denormalize(df_res['pred_4'], temp_mean, temp_std)
    df_res['pred_8'] = denormalize(df_res['pred_8'], temp_mean, temp_std)
    df_res['pred_12'] = denormalize(df_res['pred_12'], temp_mean, temp_std)
    
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(12,6))
    plt.plot(df_res['time'], df_res['real'], label='Valor real')
    plt.plot(df_res['time'], df_res['pred_4'], label='Predicción 4h')
    plt.plot(df_res['time'], df_res['pred_8'], label='Predicción 8h')
    plt.plot(df_res['time'], df_res['pred_12'], label='Predicción 12h')
    plt.xlabel('Tiempo')
    plt.ylabel(OUTPUT_FEATURE)
    plt.title(f'Comparación de predicciones para {OUTPUT_FEATURE}')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/temperature_predictions.png")
    plt.close()
    
    df_res.to_csv("results/temperature_predictions.csv", index=False)
    
    print("Proceso de predicción y cotejo completado. Resultados guardados en la carpeta 'results'.")

if __name__ == '__main__':
    main()
