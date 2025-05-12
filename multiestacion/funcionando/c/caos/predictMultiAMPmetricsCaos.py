import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import timedelta
from functools import reduce
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score, roc_auc_score

# -------------------------
# Funciones auxiliares y preprocesado
# -------------------------
def smape(y_true, y_pred):
    denominator = np.abs(y_true) + np.abs(y_pred)
    diff = np.abs(y_pred - y_true)
    return 100 * np.mean(np.where(denominator == 0, 0, 2 * diff / denominator))

def add_temporal_features(df, time_col='time'):
    df[time_col] = pd.to_datetime(df[time_col])
    df['hour'] = df[time_col].dt.hour
    df['day_of_week'] = df[time_col].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    return df

def load_station_data(station, filename, base_features):
    # Se asume que se deben saltar las dos primeras filas (metadata)
    df = pd.read_csv(filename, skiprows=2, parse_dates=['time'])
    df.sort_values('time', inplace=True)
    df = add_temporal_features(df, time_col='time')
    if station != 'galapagar':
        rename_dict = {col: f"{station}_{col}" for col in base_features}
        df.rename(columns=rename_dict, inplace=True)
    return df

# -------------------------
# Definición del Modelo: ForecastNet
# -------------------------
class ForecastNet(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, output_window, output_dim, dropout=0.0):
        """
        Modelo basado en LSTM para series temporales.
        """
        super(ForecastNet, self).__init__()
        self.encoder = nn.LSTM(input_size=num_features, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        self.decoder = nn.LSTM(input_size=num_features, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_dim)
        self.proj = nn.Linear(output_dim, num_features)
        self.output_window = output_window

    def forward(self, src):
        # Codificación
        _, (hidden, cell) = self.encoder(src)
        # Para el decoder se usan las variables de Galapagar (se asume que temperatura y lluvia son columnas 0 y 2)
        decoder_input = src[:, -1:, :][:, :, [0, 2]]  # forma: [batch, 1, 2]
        outputs = []
        for t in range(self.output_window):
            decoder_input_proj = self.proj(decoder_input)  # forma: [batch, 1, num_features]
            out, (hidden, cell) = self.decoder(decoder_input_proj, (hidden, cell))
            pred = self.fc(out)  # forma: [batch, 1, output_dim]
            outputs.append(pred)
            decoder_input = pred  # retroalimentación
        outputs = torch.cat(outputs, dim=1)
        return outputs

# -------------------------
# Función para actualizar un renglón de la ventana de entrada
# -------------------------
def update_new_row(new_time, base_row, predicted_vals, input_features):
    """
    Actualiza un renglón base (vector de tamaño num_features) para la nueva hora.
    - En la parte de Galapagar se actualizan 'temperature_2m (°C)' y 'rain (mm)' con predicted_vals.
    - Se recalculan las variables temporales (para cualquier columna que termine con 'hour_sin', 'hour_cos', etc.)
    
    Args:
        new_time: pd.Timestamp de la nueva hora.
        base_row: tensor 1D de tamaño [num_features] (los valores del último renglón de la ventana).
        predicted_vals: array con 2 valores (predicción para temperatura y lluvia).
        input_features: lista de nombres de columnas (en el mismo orden que base_row).
        
    Retorna:
        Un tensor actualizado (1D) de tamaño [num_features].
    """
    new_row = base_row.clone()
    for idx, col in enumerate(input_features):
        # Actualizar la parte de Galapagar: se asume que las columnas sin prefijo corresponden a Galapagar
        if col == "temperature_2m (°C)":
            new_row[idx] = float(predicted_vals[0])
        elif col == "rain (mm)":
            new_row[idx] = float(predicted_vals[1])
        # Actualizar variables temporales basadas en new_time:
        if col.endswith("hour_sin"):
            new_row[idx] = math.sin(2 * math.pi * new_time.hour / 24)
        elif col.endswith("hour_cos"):
            new_row[idx] = math.cos(2 * math.pi * new_time.hour / 24)
        elif col.endswith("dow_sin"):
            new_row[idx] = math.sin(2 * math.pi * new_time.dayofweek / 7)
        elif col.endswith("dow_cos"):
            new_row[idx] = math.cos(2 * math.pi * new_time.dayofweek / 7)
    return new_row

# -------------------------
# Predicción recursiva adaptada (iterativa por hora dentro de cada bloque)
# -------------------------
def recursive_prediction(model, x_input, forecast_start, total_steps=72, step_size=24, input_features=None):
    """
    Realiza predicción recursiva para alcanzar un horizonte total de 'total_steps' horas,
    usando el modelo (entrenado para predecir bloques de 'step_size' horas).
    
    Para cada bloque, se obtiene la predicción (forma: [1, step_size, 2]) y se actualiza la ventana
    de entrada construyendo nuevos renglones que:
      - Reemplazan la temperatura y lluvia de Galapagar con los valores predichos.
      - Recalculan las variables temporales para la hora correspondiente.
      - Para el resto de las variables se conserva el último valor conocido.
      
    Args:
        model: Modelo entrenado (ForecastNet).
        x_input: Tensor de entrada con forma [1, INPUT_WINDOW, num_features].
        forecast_start: pd.Timestamp con la hora inicial de la predicción.
        total_steps: Horizonte total deseado (ej. 72 horas).
        step_size: Horizonte del modelo (ej. 24 horas).
        input_features: Lista de nombres de columnas en el mismo orden que x_input.
        
    Retorna:
        Tensor con la predicción concatenada (forma: [1, total_steps, output_dim]).
    """
    all_preds = []
    current_input = x_input.clone()
    # Contador de horas predichas (para actualizar la hora en cada nuevo renglón)
    counter = 0  
    num_iterations = total_steps // step_size

    for i in range(num_iterations):
        with torch.no_grad():
            # Predicción para el bloque actual (24 horas)
            pred_block = model(current_input)  # forma: [1, step_size, output_dim] (output_dim = 2)
        all_preds.append(pred_block)
        # Convertir bloque de predicción a NumPy para facilidad en el loop (forma: [step_size, 2])
        pred_block_np = pred_block.squeeze(0).cpu().numpy()
        new_rows = []
        # Para cada hora en el bloque, se genera un nuevo renglón para la ventana
        for t in range(step_size):
            counter += 1
            new_time = forecast_start + pd.Timedelta(hours=counter)
            # Tomar el último renglón de la ventana actual como base
            base_row = current_input[0, -1, :]
            new_row = update_new_row(new_time, base_row, pred_block_np[t], input_features)
            # new_row tiene forma [num_features]; se agrega como tensor 2D (1, num_features)
            new_rows.append(new_row.unsqueeze(0))
        # Formar un bloque nuevo con las filas generadas: forma [1, step_size, num_features]
        new_block = torch.cat(new_rows, dim=0).unsqueeze(0)
        # Actualizar la ventana: se eliminan los primeros 'step_size' renglones y se agregan los nuevos al final
        current_input = torch.cat([current_input[:, step_size:, :], new_block], dim=1)
    # Concatenar todas las predicciones en el eje del tiempo: forma [1, total_steps, output_dim]
    return torch.cat(all_preds, dim=1)

# -------------------------
# Ensemble de sensibilidad con perturbaciones y predicción recursiva
# -------------------------
def ensemble_sensitivity_recursive(model, df_norm, input_features, forecast_date_str="2025-02-01 00:00:00",
                                   ensemble_window_hours=72, total_pred_hours=72, step_size=24):
    """
    Realiza 100 predicciones recursivas (hasta total_pred_hours) variando la ventana de entrada en ±0.5%
    de forma uniforme para todas las variables, y plotea el ensemble de la predicción de temperatura.
    
    Args:
        model: Modelo entrenado (ForecastNet) (configurado para step_size horas).
        df_norm: DataFrame de datos normalizados (con índice 'time').
        input_features: Lista de nombres de columnas (en el mismo orden que se usó para entrenar).
        forecast_date_str: Fecha inicial de la predicción (string, ej. "2025-02-01 00:00:00").
        ensemble_window_hours: Número de horas de la ventana de entrada.
        total_pred_hours: Horizonte total deseado (ej. 72 horas).
        step_size: Horizonte base del modelo (ej. 24 horas).
    """
    forecast_date = pd.Timestamp(forecast_date_str)
    window_start = forecast_date - pd.Timedelta(hours=ensemble_window_hours)
    
    # Extraer la ventana de entrada (asegúrate de que df_norm tenga datos para las fechas requeridas)
    try:
        window_data = df_norm.loc[window_start: forecast_date - pd.Timedelta(hours=1), input_features].values
    except KeyError:
        raise KeyError("No se encontraron datos en la ventana especificada. Verifica las fechas en df_norm.")
        
    if window_data.shape[0] != ensemble_window_hours:
        raise ValueError(f"La ventana de entrada debe tener {ensemble_window_hours} registros, pero tiene {window_data.shape[0]}.")

    # Generar 100 factores de perturbación de -0.5% a 0.5%
    perturbations = np.linspace(-0.05, 0.05, 100)
    ensemble_predictions = []

    for factor in perturbations:
        # Perturbar de forma uniforme la ventana de entrada
        window_data_perturbed = window_data.copy() * (1 + factor)
        x_input = torch.tensor(window_data_perturbed, dtype=torch.float32).unsqueeze(0).to(next(model.parameters()).device)
        
        # Realizar predicción recursiva para obtener total_pred_hours (ej. 72 horas)
        pred_recursive = recursive_prediction(model, x_input, forecast_start=forecast_date,
                                               total_steps=total_pred_hours, step_size=step_size,
                                               input_features=input_features)
        # Extraer la predicción de temperatura (se asume que es el primer elemento de la salida)
        pred_temperature = pred_recursive.squeeze(0).cpu().numpy()[:, 0]  # forma: [total_pred_hours]
        ensemble_predictions.append(pred_temperature)
    
    ensemble_predictions = np.array(ensemble_predictions)  # forma: [100, total_pred_hours]
    
    # Plot del ensemble: cada curva corresponde a una perturbación
    plt.figure(figsize=(12, 6))
    horizon_hours = np.arange(1, total_pred_hours + 1)
    for pred in ensemble_predictions:
        plt.plot(horizon_hours, pred, color='blue', alpha=0.3)
    
    # Graficar la media del ensemble
    mean_prediction = ensemble_predictions.mean(axis=0)
    plt.plot(horizon_hours, mean_prediction, color='red', linewidth=2, label="Media del Ensemble")
    
    plt.xlabel("Horas de Predicción")
    plt.ylabel("Temperatura Predicha (normalizada)")
    plt.title(f"Ensemble de Predicciones de Temperatura a {total_pred_hours}h\n"
              f"(Ventana de entrada perturbada ±0.5% en todas las variables)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -------------------------
# Parámetros y configuración global
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo utilizado:", device, flush=True)

# Parámetros de predicción y datos
INPUT_WINDOW = 74    # Últimas 74 horas para formar la ventana de entrada
OUTPUT_WINDOW = 24   # Horizonte base del modelo (24 horas)
total_prediction_hours = 72  # Horizonte total deseado en recursividad

FEATURES = ['temperature_2m (°C)', 'relative_humidity_2m (%)', 'rain (mm)',
            'surface_pressure (hPa)', 'wind_speed_10m (km/h)', 'wind_direction_10m (°)']
TEMPORAL_FEATURES = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
BASE_FEATURES = FEATURES + TEMPORAL_FEATURES  # 10 features por estación
OUTPUT_FEATURES = ['temperature_2m (°C)', 'rain (mm)']  # Se predicen temperatura y lluvia para Galapagar

HIDDEN_SIZE = 64
NUM_LAYERS = 2

# Diccionario de archivos con datos de estaciones
station_files = {
    'galapagar': 'galapagarhoraria25.csv',
    'santander': 'santander25.csv',
    'pontevedra': 'pontevedra25.csv',
    'salamanca': 'salamanca25.csv',
    'huelva': 'huelva25.csv',
    'ciudadreal': 'ciudadreal25.csv',
    'valencia': 'valencia25.csv',
    'almeria': 'almeria25.csv',
    'creus': 'creus25.csv',
    'guadalajara': 'guadalajara25.csv',
    'burgos': 'burgos25.csv'
}

# Construir la lista de features de entrada para todas las estaciones
input_features = []
for station in station_files.keys():
    if station == 'galapagar':
        input_features += BASE_FEATURES
    else:
        input_features += [f"{station}_{col}" for col in BASE_FEATURES]

# -------------------------
# Función principal
# -------------------------
def main():
    # Cargar y fusionar los datos de cada estación
    data_frames = []
    for station, file in station_files.items():
        if not os.path.exists(file):
            raise FileNotFoundError(f"El archivo {file} no se encontró.")
        df_station = load_station_data(station, file, BASE_FEATURES)
        if station == 'galapagar':
            cols = ['time'] + BASE_FEATURES
        else:
            cols = ['time'] + [f"{station}_{col}" for col in BASE_FEATURES]
        data_frames.append(df_station[cols])
    
    df_merged = reduce(lambda left, right: pd.merge(left, right, on='time', how='inner'), data_frames)
    df_merged.sort_values('time', inplace=True)
    print(f"Total de registros en la unión: {df_merged.shape[0]}")
    
    # Cargar estadísticas de normalización
    stats_df = pd.read_csv("model/normalization_stats.csv", index_col=0)
    if stats_df.shape[1] != 2 and stats_df.shape[0] == 2:
        stats_df = stats_df.T
    if list(stats_df.columns) != ['mean', 'std']:
        stats_df.columns = ['mean', 'std']
    
    norm_stats = {}
    for col in input_features:
        try:
            norm_stats[col] = (float(stats_df.loc[col, 'mean']), float(stats_df.loc[col, 'std']))
        except KeyError:
            raise KeyError(f"La variable {col} no se encontró en normalization_stats.csv")
    
    # Normalizar los datos
    df_norm = df_merged.copy()
    for col in input_features:
        mean, std = norm_stats[col]
        df_norm[col] = (df_norm[col] - mean) / std
    df_norm.set_index('time', inplace=True)
    
    # Cargar el modelo entrenado
    model = ForecastNet(num_features=len(input_features), hidden_size=HIDDEN_SIZE,
                        num_layers=NUM_LAYERS, output_window=OUTPUT_WINDOW,
                        output_dim=len(OUTPUT_FEATURES), dropout=0.2)
    if not os.path.exists("model/forecast_model.pt"):
        raise FileNotFoundError("El archivo model/forecast_model.pt no se encontró.")
    # Se recomienda para mayor seguridad usar weights_only=True si es posible.
    model.load_state_dict(torch.load("model/forecast_model.pt", map_location=device))
    model = model.to(device)
    model.eval()
    
    # Ejecutar el ensemble de sensibilidad con recursividad para el 1 de febrero de 2025
    ensemble_sensitivity_recursive(model, df_norm, input_features,
                                   forecast_date_str="2025-02-01 00:00:00",
                                   ensemble_window_hours=72,
                                   total_pred_hours=total_prediction_hours,
                                   step_size=OUTPUT_WINDOW)

if __name__ == '__main__':
    main()
