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

# Lista de características (usar ALL_FEATURES: 6 originales + 4 temporales)
FEATURES = ['temperature_2m (°C)', 'relative_humidity_2m (%)', 'rain (mm)',
            'pressure_msl (hPa)', 'wind_speed_10m (km/h)', 'wind_direction_10m (°)']
TEMPORAL_FEATURES = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
ALL_FEATURES = FEATURES + TEMPORAL_FEATURES

# Variables objetivo: temperatura y lluvia
OUTPUT_FEATURES = ['temperature_2m (°C)', 'rain (mm)']

HIDDEN_SIZE = 64
NUM_LAYERS = 2

# Modelo (adaptado para predecir dos variables)
class ForecastNet(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, output_window, output_dim, dropout=0.0):
        """
        Args:
            num_features: Número total de features de entrada.
            hidden_size: Tamaño de la capa oculta.
            num_layers: Número de capas en el LSTM.
            output_window: Número de pasos de tiempo a predecir.
            output_dim: Dimensión de la salida (2 para [temperatura, lluvia]).
            dropout: Tasa de dropout.
        """
        super(ForecastNet, self).__init__()
        self.encoder = nn.LSTM(input_size=num_features, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        self.decoder = nn.LSTM(input_size=num_features, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        # Capa final: proyecta desde hidden_size a output_dim (2 variables)
        self.fc = nn.Linear(hidden_size, output_dim)
        # Proyección para adaptar la dimensión de la entrada del decoder a num_features
        self.proj = nn.Linear(output_dim, num_features)
        self.output_window = output_window

    def forward(self, src):
        """
        Args:
            src: Tensor de entrada de forma [batch, INPUT_WINDOW, num_features].
        Returns:
            outputs: Tensor de predicción de forma [batch, OUTPUT_WINDOW, output_dim].
        """
        _, (hidden, cell) = self.encoder(src)
        # Inicialización del decoder: se usan las columnas correspondientes a temperatura (índice 0)
        # y lluvia (índice 2) de la última fila de la secuencia de entrada.
        decoder_input = src[:, -1:, :][:, :, [0, 2]]  # [batch, 1, 2]
        outputs = []
        for t in range(self.output_window):
            decoder_input_proj = self.proj(decoder_input)  # [batch, 1, num_features]
            out, (hidden, cell) = self.decoder(decoder_input_proj, (hidden, cell))
            pred = self.fc(out)  # [batch, 1, output_dim]
            outputs.append(pred)
            # Retroalimentación: se usa la predicción actual como siguiente entrada del decoder.
            decoder_input = pred
        outputs = torch.cat(outputs, dim=1)  # [batch, OUTPUT_WINDOW, output_dim]
        return outputs

# Función para desnormalizar una serie dado su media y desviación estándar
def denormalize(series, mean, std):
    return series * std + mean

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
    
    # Calcula end_pred usando la columna 'time' antes de establecerla como índice
    end_pred = df['time'].max() - pd.Timedelta(hours=OUTPUT_WINDOW)
    
    # Agregar características temporales
    df = add_temporal_features(df, time_col='time')
    
    # Cargar parámetros de normalización
    stats_df = pd.read_csv("model/normalization_stats.csv", index_col=0)
    if stats_df.shape[1] != 2 and stats_df.shape[0] == 2:
        stats_df = stats_df.T
    if stats_df.shape[1] != 2:
        raise ValueError(f"El archivo normalization_stats.csv tiene un formato inesperado: {stats_df.shape}")
    if list(stats_df.columns) != ['mean', 'std']:
        stats_df.columns = ['mean', 'std']
    
    # Se extraen medias y desviaciones para cada variable de ALL_FEATURES
    norm_stats = {}
    for col in ALL_FEATURES:
        try:
            norm_stats[col] = (float(stats_df.loc[col, 'mean']), float(stats_df.loc[col, 'std']))
        except KeyError:
            raise KeyError(f"La variable {col} no se encontró en normalization_stats.csv")
    
    # Medias y std para las variables objetivo (temperatura y lluvia)
    temp_mean, temp_std = norm_stats[OUTPUT_FEATURES[0]]
    rain_mean, rain_std = norm_stats[OUTPUT_FEATURES[1]]
    
    # Normalizar los datos de predicción usando ALL_FEATURES
    df_norm = df.copy()
    for col in ALL_FEATURES:
        mean, std = norm_stats[col]
        df_norm[col] = (df_norm[col] - mean) / std
    
    # Establecer 'time' como índice
    df_norm.set_index('time', inplace=True)
    
    # Cargar el modelo entrenado (se espera que se entrenó con ALL_FEATURES y 2 variables de salida)
    model = ForecastNet(num_features=len(ALL_FEATURES), hidden_size=HIDDEN_SIZE,
                        num_layers=NUM_LAYERS, output_window=OUTPUT_WINDOW,
                        output_dim=len(OUTPUT_FEATURES), dropout=0.2)
    # Nota: En versiones recientes se omite el argumento weights_only. Se carga el estado del modelo.
    model.load_state_dict(torch.load("model/forecast_model.pt", map_location=device))
    model = model.to(device)
    model.eval()
    
    # Generar predicciones a partir de una fecha de inicio definida hasta end_pred
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
            # La predicción ahora tiene forma [1, OUTPUT_WINDOW, 2] (temperatura y lluvia)
            pred = model(x_input)
        pred = pred.squeeze(0).cpu().numpy()  # [OUTPUT_WINDOW, 2]
        forecast_dict[current_time] = pred
        current_time += pd.Timedelta(hours=1)
    
    # Comparar las predicciones con valores reales a partir de una fecha de evaluación
    eval_start = pd.Timestamp("2025-01-11 00:00:00")
    results = []
    current_time = eval_start
    while current_time in df_norm.index:
        try:
            # Valores reales de temperatura y lluvia (se toman de df_norm, que está normalizado)
            real_temp = df_norm.loc[current_time, OUTPUT_FEATURES[0]]
            real_rain = df_norm.loc[current_time, OUTPUT_FEATURES[1]]
            # Extraer las predicciones: se toman ventanas que iniciaron 4, 8 y 12 horas antes respectivamente
            forecast_temp_4 = forecast_dict[current_time - pd.Timedelta(hours=4)][3][0]
            forecast_temp_8 = forecast_dict[current_time - pd.Timedelta(hours=8)][7][0]
            forecast_temp_12 = forecast_dict[current_time - pd.Timedelta(hours=12)][11][0]
            forecast_rain_4 = forecast_dict[current_time - pd.Timedelta(hours=4)][3][1]
            forecast_rain_8 = forecast_dict[current_time - pd.Timedelta(hours=8)][7][1]
            forecast_rain_12 = forecast_dict[current_time - pd.Timedelta(hours=12)][11][1]
        except KeyError:
            current_time += pd.Timedelta(hours=1)
            continue
        
        results.append({
            'time': current_time,
            'real_temp': real_temp,
            'pred_temp_4': forecast_temp_4,
            'pred_temp_8': forecast_temp_8,
            'pred_temp_12': forecast_temp_12,
            'real_rain': real_rain,
            'pred_rain_4': forecast_rain_4,
            'pred_rain_8': forecast_rain_8,
            'pred_rain_12': forecast_rain_12
        })
        current_time += pd.Timedelta(hours=1)
    
    df_res = pd.DataFrame(results)
    # Desnormalizar las variables de temperatura y lluvia
    df_res['real_temp'] = denormalize(df_res['real_temp'], temp_mean, temp_std)
    df_res['pred_temp_4'] = denormalize(df_res['pred_temp_4'], temp_mean, temp_std)
    df_res['pred_temp_8'] = denormalize(df_res['pred_temp_8'], temp_mean, temp_std)
    df_res['pred_temp_12'] = denormalize(df_res['pred_temp_12'], temp_mean, temp_std)
    
    df_res['real_rain'] = denormalize(df_res['real_rain'], rain_mean, rain_std)
    df_res['pred_rain_4'] = denormalize(df_res['pred_rain_4'], rain_mean, rain_std)
    df_res['pred_rain_8'] = denormalize(df_res['pred_rain_8'], rain_mean, rain_std)
    df_res['pred_rain_12'] = denormalize(df_res['pred_rain_12'], rain_mean, rain_std)
    
    os.makedirs("results", exist_ok=True)
    
    # Gráfico de temperatura
    plt.figure(figsize=(12,6))
    plt.plot(df_res['time'], df_res['real_temp'], label='Temperatura real')
    plt.plot(df_res['time'], df_res['pred_temp_4'], label='Predicción 4h')
    plt.plot(df_res['time'], df_res['pred_temp_8'], label='Predicción 8h')
    plt.plot(df_res['time'], df_res['pred_temp_12'], label='Predicción 12h')
    plt.xlabel('Tiempo')
    plt.ylabel('Temperatura (°C)')
    plt.title('Comparación de predicciones de Temperatura')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/temperature_predictions.png")
    plt.close()
    
    # Gráfico de lluvia
    plt.figure(figsize=(12,6))
    plt.plot(df_res['time'], df_res['real_rain'], label='Lluvia real')
    plt.plot(df_res['time'], df_res['pred_rain_4'], label='Predicción 4h')
    plt.plot(df_res['time'], df_res['pred_rain_8'], label='Predicción 8h')
    plt.plot(df_res['time'], df_res['pred_rain_12'], label='Predicción 12h')
    plt.xlabel('Tiempo')
    plt.ylabel('Lluvia (mm)')
    plt.title('Comparación de predicciones de Lluvia')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/rain_predictions.png")
    plt.close()
    
    df_res.to_csv("results/temperature_rain_predictions.csv", index=False)
    print("Proceso de predicción y cotejo completado. Resultados guardados en la carpeta 'results'.")
    print("Fecha mínima:", df_merged['time'].min())
    print("Fecha máxima:", df_merged['time'].max())

if __name__ == '__main__':
    main()
