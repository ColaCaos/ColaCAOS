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
OUTPUT_WINDOW = 72    # Ventana de salida: 72 horas de predicción
FEATURES = ['temperature_2m (°C)', 'relative_humidity_2m (%)', 'rain (mm)', 
            'pressure_msl (hPa)', 'wind_speed_10m (km/h)', 'wind_direction_10m (°)']
HIDDEN_SIZE = 64
NUM_LAYERS = 2

# Definición del modelo (mismo que en entrenamiento)
class ForecastNet(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, output_window):
        super(ForecastNet, self).__init__()
        self.encoder = nn.LSTM(input_size=num_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(input_size=num_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_features)
        self.output_window = output_window
        
    def forward(self, src):
        _, (hidden, cell) = self.encoder(src)
        decoder_input = src[:, -1:, :]  # Último valor de la secuencia de entrada
        outputs = []
        for t in range(self.output_window):
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            pred = self.fc(out)
            outputs.append(pred)
            decoder_input = pred  # feeding autoregresivo
        outputs = torch.cat(outputs, dim=1)
        return outputs

# Función para desnormalizar usando los parámetros guardados
def denormalize(series, mean, std):
    return series * std + mean

def main():
    # Cargar datos de 2025
    data_path = 'galapagarhoraria25.csv'
    df = pd.read_csv(data_path, skiprows=2, parse_dates=['time'])
    df.sort_values('time', inplace=True)
    
    # Cargar parámetros de normalización
    # Se lee el CSV una sola vez
    stats_df = pd.read_csv("model/normalization_stats.csv", index_col=0)
    # Revisar la forma del DataFrame para determinar si es necesario transponer
    # Se espera que tenga 6 filas (una por variable) y 2 columnas (mean, std)
    if stats_df.shape[1] != 2 and stats_df.shape[0] == 2:
        stats_df = stats_df.T
    if stats_df.shape[1] != 2:
        raise ValueError(f"El archivo normalization_stats.csv tiene un formato inesperado: {stats_df.shape}")
    # Renombrar las columnas si no se llaman 'mean' y 'std'
    if list(stats_df.columns) != ['mean', 'std']:
        stats_df.columns = ['mean', 'std']
    
    # Preparar el diccionario de normalización
    normalization_stats = {}
    for col in FEATURES:
        try:
            mean = float(stats_df.loc[col, 'mean'])
            std = float(stats_df.loc[col, 'std'])
        except KeyError:
            raise KeyError(f"La variable {col} no se encontró en normalization_stats.csv")
        normalization_stats[col] = (mean, std)
    
    # Normalizar datos de 2025
    df_norm = df.copy()
    for col in FEATURES:
        mean, std = normalization_stats[col]
        df_norm[col] = (df_norm[col] - mean) / std
    
    # Cargar modelo entrenado
    model = ForecastNet(num_features=len(FEATURES), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_window=OUTPUT_WINDOW)
    model.load_state_dict(torch.load("model/forecast_model.pt", map_location=device))
    model = model.to(device)
    model.eval()
    
    # Generar predicciones horarias a partir del 8 de enero de 2025
    start_pred = pd.Timestamp("2025-01-08 00:00:00")
    end_pred = df_norm['time'].max() - pd.Timedelta(hours=OUTPUT_WINDOW)
    
    # Diccionario para almacenar las predicciones:
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
            pred = model(x_input)  # forma: [1, 72, num_features]
        pred = pred.squeeze(0).cpu().numpy()  # forma: [72, num_features]
        forecast_dict[current_time] = pred
        current_time += pd.Timedelta(hours=1)
    
    # Cotejo de predicciones con valores reales:
    # Se evalúa a partir del 11 de enero de 2025, a las 00:00 horas.
    eval_start = pd.Timestamp("2025-01-11 00:00:00")
    eval_end = df_norm.index.max()
    
    # Se guardarán resultados por variable
    results = {feat: [] for feat in FEATURES}
    times = []
    
    # Para cada instante de evaluación T a partir de eval_start, se extraen:
    # - predicción hecha 4 horas antes: forecast_dict[T - 4h] índice 3
    # - predicción hecha 8 horas antes: forecast_dict[T - 8h] índice 7
    # - predicción hecha 12 horas antes: forecast_dict[T - 12h] índice 11
    current_time = eval_start
    while current_time in df_norm.index:
        try:
            real = df_norm.loc[current_time, FEATURES].values.astype(np.float32)
            forecast_4 = forecast_dict[current_time - pd.Timedelta(hours=4)][3]
            forecast_8 = forecast_dict[current_time - pd.Timedelta(hours=8)][7]
            forecast_12 = forecast_dict[current_time - pd.Timedelta(hours=12)][11]
        except KeyError:
            current_time += pd.Timedelta(hours=1)
            continue
        
        times.append(current_time)
        for i, feat in enumerate(FEATURES):
            results[feat].append({
                'time': current_time,
                'real': real[i],
                'pred_4': forecast_4[i],
                'pred_8': forecast_8[i],
                'pred_12': forecast_12[i]
            })
        current_time += pd.Timedelta(hours=1)
    
    # Graficar y guardar CSV para cada parámetro
    os.makedirs("results", exist_ok=True)
    for feat in FEATURES:
        df_res = pd.DataFrame(results[feat])
        # Desnormalizar
        mean, std = normalization_stats[feat]
        df_res['real'] = df_res['real'] * std + mean
        df_res['pred_4'] = df_res['pred_4'] * std + mean
        df_res['pred_8'] = df_res['pred_8'] * std + mean
        df_res['pred_12'] = df_res['pred_12'] * std + mean
        
        plt.figure(figsize=(12,6))
        plt.plot(df_res['time'], df_res['real'], label='Valor real')
        plt.plot(df_res['time'], df_res['pred_4'], label='Predicción 4h')
        plt.plot(df_res['time'], df_res['pred_8'], label='Predicción 8h')
        plt.plot(df_res['time'], df_res['pred_12'], label='Predicción 12h')
        plt.xlabel('Tiempo')
        plt.ylabel(feat)
        plt.title(f'Comparación de predicciones para {feat}')
        plt.legend()
        plt.tight_layout()
        safe_feat = feat.replace(' ', '_').replace('(', '').replace(')', '')
        plt.savefig(f"results/{safe_feat}_predictions.png")
        plt.close()
        
        df_res.to_csv(f"results/{safe_feat}_predictions.csv", index=False)
    
    print("Proceso de predicción y cotejo completado. Resultados guardados en la carpeta 'results'.")

if __name__ == '__main__':
    main()
