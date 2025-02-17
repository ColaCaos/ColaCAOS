import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import timedelta
import os

# Configuración de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo utilizado:", device)

# Parámetros (deben coincidir con el entrenamiento)
INPUT_WINDOW = 168    # Ventana de entrada: 7 días
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
        batch_size = src.size(0)
        _, (hidden, cell) = self.encoder(src)
        decoder_input = src[:, -1:, :]  # último valor de la secuencia de entrada
        outputs = []
        for t in range(self.output_window):
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            pred = self.fc(out)
            outputs.append(pred)
            decoder_input = pred
        outputs = torch.cat(outputs, dim=1)
        return outputs

# Función para desnormalizar usando los parámetros guardados
def denormalize(series, mean, std):
    return series * std + mean

def main():
    # Cargar datos de 2025
    data_path = 'galapagarhoraria25.csv'
    df = pd.read_csv(data_path,skiprows=2, parse_dates=['time'])
    df.sort_values('time', inplace=True)
    
    # Cargar parámetros de normalización
    stats_df = pd.read_csv("model/normalization_stats.csv", index_col=0)
    normalization_stats = {}
    for col in FEATURES:
        mean = float(stats_df.loc[0, col])
        std = float(stats_df.loc[1, col])
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
    # Se asume que la columna 'time' es de tipo datetime.
    start_pred = pd.Timestamp("2025-01-08 00:00:00")
    end_pred = df_norm['time'].max() - pd.Timedelta(hours=OUTPUT_WINDOW)
    
    # Diccionario para almacenar predicciones:
    # Se guardarán predicciones con clave: instante de inicio (fecha de predicción) y valor: vector de 72 predicciones (cada uno con 6 parámetros)
    forecast_dict = {}
    
    # Se asume que el DataFrame tiene registros horarios consecutivos
    df_norm.set_index('time', inplace=True)
    
    current_time = start_pred
    while current_time <= end_pred:
        # Ventana de entrada: 168 horas previas a current_time
        window_start = current_time - pd.Timedelta(hours=INPUT_WINDOW)
        window_data = df_norm.loc[window_start: current_time - pd.Timedelta(hours=1), FEATURES].values
        if window_data.shape[0] != INPUT_WINDOW:
            # En caso de que no existan suficientes datos, se salta
            current_time += pd.Timedelta(hours=1)
            continue
        
        # Convertir a tensor y agregar dimensión batch
        x_input = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x_input)  # forma: [1, 72, num_features]
        pred = pred.squeeze(0).cpu().numpy()  # [72, num_features]
        forecast_dict[current_time] = pred
        current_time += pd.Timedelta(hours=1)
    
    # Ahora, cotejo de predicciones con valores reales:
    # Se evaluará a partir del 11 de enero de 2025, a las 00:00 horas.
    eval_start = pd.Timestamp("2025-01-11 00:00:00")
    eval_end = df_norm.index.max()
    
    # Listas para almacenar resultados por variable
    results = {feat: [] for feat in FEATURES}
    times = []
    
    # Para cada instante de evaluación T a partir de eval_start, se extraen:
    # - predicción hecha 24 horas antes: en forecast_dict[T - 24h] en el índice 23 (corresponde a t+24)
    # - predicción hecha 48 horas antes: en forecast_dict[T - 48h] en el índice 47
    # - predicción hecha 72 horas antes: en forecast_dict[T - 72h] en el índice 71
    current_time = eval_start
    while current_time in df_norm.index:
        try:
            # Extraer valor real (normalizado)
            real = df_norm.loc[current_time, FEATURES].values.astype(np.float32)
            # Extraer las tres predicciones correspondientes
            forecast_24 = forecast_dict[current_time - pd.Timedelta(hours=24)][23]  # índice 23 para t+24
            forecast_48 = forecast_dict[current_time - pd.Timedelta(hours=48)][47]
            forecast_72 = forecast_dict[current_time - pd.Timedelta(hours=72)][71]
        except KeyError:
            # Si no se tiene predicción para alguno de los tiempos, se omite el instante
            current_time += pd.Timedelta(hours=1)
            continue
        
        times.append(current_time)
        for i, feat in enumerate(FEATURES):
            results[feat].append({
                'time': current_time,
                'real': real[i],
                'pred_24': forecast_24[i],
                'pred_48': forecast_48[i],
                'pred_72': forecast_72[i]
            })
        current_time += pd.Timedelta(hours=1)
    
    # Para cada parámetro se desnormalizan los valores, se generan gráficos y se guardan los datos en CSV
    os.makedirs("results", exist_ok=True)
    for feat in FEATURES:
        # Convertir a DataFrame
        df_res = pd.DataFrame(results[feat])
        # Desnormalizar
        mean, std = normalization_stats[feat]
        df_res['real'] = df_res['real'] * std + mean
        df_res['pred_24'] = df_res['pred_24'] * std + mean
        df_res['pred_48'] = df_res['pred_48'] * std + mean
        df_res['pred_72'] = df_res['pred_72'] * std + mean
        
        # Graficar
        plt.figure(figsize=(12,6))
        plt.plot(df_res['time'], df_res['real'], label='Valor real')
        plt.plot(df_res['time'], df_res['pred_24'], label='Predicción 24h')
        plt.plot(df_res['time'], df_res['pred_48'], label='Predicción 48h')
        plt.plot(df_res['time'], df_res['pred_72'], label='Predicción 72h')
        plt.xlabel('Tiempo')
        plt.ylabel(feat)
        plt.title(f'Comparación de predicciones para {feat}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/{feat.replace(' ', '_').replace('(', '').replace(')', '')}_predictions.png")
        plt.close()
        
        # Guardar CSV
        df_res.to_csv(f"results/{feat.replace(' ', '_').replace('(', '').replace(')', '')}_predictions.csv", index=False)
    
    print("Proceso de predicción y cotejo completado. Los resultados se han guardado en la carpeta 'results'.")

if __name__ == '__main__':
    main()
