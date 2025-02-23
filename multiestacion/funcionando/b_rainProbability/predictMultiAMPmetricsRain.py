import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import timedelta
from functools import reduce
import math

# Importar métricas de clasificación
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score, roc_auc_score

# -------------------------
# Función para calcular SMAPE
# -------------------------
def smape(y_true, y_pred):
    denominator = np.abs(y_true) + np.abs(y_pred)
    diff = np.abs(y_pred - y_true)
    return 100 * np.mean(np.where(denominator == 0, 0, 2 * diff / denominator))

# -------------------------
# Función de normalización (global)
# -------------------------
def normalize_data(df, feature_cols):
    stats = {}
    df_norm = df.copy()
    for col in feature_cols:
        mean = df_norm[col].mean()
        std = df_norm[col].std()
        df_norm[col] = (df_norm[col] - mean) / std
        stats[col] = (mean, std)
    return df_norm, stats

# -------------------------
# Funciones de preprocesado
# -------------------------
def add_temporal_features(df, time_col='time'):
    df[time_col] = pd.to_datetime(df[time_col])
    df['hour'] = df[time_col].dt.hour
    df['day_of_week'] = df[time_col].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    return df

def load_station_data(station, filename):
    # Se asume que se deben saltar las dos primeras filas (metadata)
    df = pd.read_csv(filename, skiprows=2, parse_dates=['time'])
    df.sort_values('time', inplace=True)
    df = add_temporal_features(df, time_col='time')
    # Para estaciones distintas a galapagar, agregar prefijo a cada feature
    if station != 'galapagar':
        rename_dict = {col: f"{station}_{col}" for col in BASE_FEATURES}
        df.rename(columns=rename_dict, inplace=True)
    return df

# -------------------------
# Configuraciones y variables globales
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo utilizado:", device, flush=True)

# Parámetros de predicción
INPUT_WINDOW = 74    # últimas 74 horas
OUTPUT_WINDOW = 24   # próximas 24 horas

FEATURES = ['temperature_2m (°C)', 'relative_humidity_2m (%)', 'rain (mm)',
            'surface_pressure (hPa)', 'wind_speed_10m (km/h)', 'wind_direction_10m (°)']
TEMPORAL_FEATURES = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
BASE_FEATURES = FEATURES + TEMPORAL_FEATURES  # 10 features por estación

# Variables objetivo: se predicen temperatura y lluvia para Galapagar.
OUTPUT_FEATURES = ['temperature_2m (°C)', 'rain (mm)']

HIDDEN_SIZE = 64
NUM_LAYERS = 2

# Diccionario de archivos (asegúrate de incluir las estaciones usadas en el entrenamiento)
station_files = {
    'galapagar': 'galapagarhoraria25.csv',
    'santander': 'santander25.csv',
    'segovia': 'segovia25.csv',
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

# -------------------------
# Modelo: ForecastNetMT (Multi-Tarea)
# -------------------------
class ForecastNetMT(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, output_window, output_dim, dropout=0.0):
        """
        Args:
            num_features: Número total de features de entrada.
            hidden_size: Tamaño de la capa oculta.
            num_layers: Número de capas en el LSTM.
            output_window: Número de pasos de tiempo a predecir.
            output_dim: Dimensión de la salida para targets principales (temperatura y lluvia regresión).
            dropout: Tasa de dropout.
        """
        super(ForecastNetMT, self).__init__()
        self.encoder = nn.LSTM(input_size=num_features, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        self.decoder = nn.LSTM(input_size=num_features, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        # Salida principal para temperatura y lluvia (regresión)
        self.fc = nn.Linear(hidden_size, output_dim)
        # Rama auxiliar para clasificación de lluvia (logits)
        self.cls_fc = nn.Linear(hidden_size, 1)
        self.proj = nn.Linear(output_dim, num_features)
        self.output_window = output_window

    def forward(self, src):
        """
        Args:
            src: Tensor de entrada de forma [batch, INPUT_WINDOW, num_features].
        Returns:
            outputs: Tensor de regresión de forma [batch, OUTPUT_WINDOW, 2] (temperatura, lluvia).
            cls_outputs: Tensor de logits de clasificación [batch, OUTPUT_WINDOW, 1].
        """
        _, (hidden, cell) = self.encoder(src)
        # Se asume que en el vector de entrada, las features de Galapagar están en el orden de BASE_FEATURES:
        # temperatura en posición 0 y lluvia en posición 2.
        decoder_input = src[:, -1:, :][:, :, [0, 2]]  # [batch, 1, 2]
        outputs = []
        cls_outputs = []
        for t in range(self.output_window):
            decoder_input_proj = self.proj(decoder_input)  # [batch, 1, num_features]
            out, (hidden, cell) = self.decoder(decoder_input_proj, (hidden, cell))
            pred = self.fc(out)  # [batch, 1, 2]
            cls_logit = self.cls_fc(out)  # [batch, 1, 1]
            outputs.append(pred)
            cls_outputs.append(cls_logit)
            decoder_input = pred  # retroalimentación
        outputs = torch.cat(outputs, dim=1)       # [batch, OUTPUT_WINDOW, 2]
        cls_outputs = torch.cat(cls_outputs, dim=1) # [batch, OUTPUT_WINDOW, 1]
        return outputs, cls_outputs

# -------------------------
# Código de predicción y evaluación
# -------------------------
def main():
    # Cargar los datos de cada estación y fusionarlos por 'time'
    data_frames = []
    for station, file in station_files.items():
        if not os.path.exists(file):
            raise FileNotFoundError(f"El archivo {file} no se encontró.")
        df_station = load_station_data(station, file)
        if station == 'galapagar':
            cols = ['time'] + BASE_FEATURES
        else:
            cols = ['time'] + [f"{station}_{col}" for col in BASE_FEATURES]
        data_frames.append(df_station[cols])
    
    df_merged = reduce(lambda left, right: pd.merge(left, right, on='time', how='inner'), data_frames)
    df_merged.sort_values('time', inplace=True)
    print(f"Total de registros en la unión: {df_merged.shape[0]}")
    
    # Construir la lista de features de entrada
    input_features = []
    for station in station_files.keys():
        if station == 'galapagar':
            input_features += BASE_FEATURES
        else:
            input_features += [f"{station}_{col}" for col in BASE_FEATURES]
    target_features = OUTPUT_FEATURES  # Se predicen solo las variables de Galapagar
    
    # Cargar las estadísticas de normalización generadas en el entrenamiento
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
    
    # Extraer estadísticas para las variables objetivo (Galapagar)
    temp_mean, temp_std = norm_stats[OUTPUT_FEATURES[0]]
    rain_mean, rain_std = norm_stats[OUTPUT_FEATURES[1]]
    
    # Normalizar los datos de predicción
    df_norm = df_merged.copy()
    for col in input_features:
        mean, std = norm_stats[col]
        df_norm[col] = (df_norm[col] - mean) / std
    df_norm.set_index('time', inplace=True)
    
    # Cargar el modelo entrenado (modelo multi-tarea)
    model = ForecastNetMT(num_features=len(input_features), hidden_size=HIDDEN_SIZE,
                          num_layers=NUM_LAYERS, output_window=OUTPUT_WINDOW,
                          output_dim=len(OUTPUT_FEATURES), dropout=0.2)
    model.load_state_dict(torch.load("model/forecast_model_mt.pt", map_location=device))
    model = model.to(device)
    model.eval()
    
    # Determinar el último instante para el que se puede hacer predicción
    end_pred = df_norm.index.max() - pd.Timedelta(hours=OUTPUT_WINDOW)
    
    # Generar predicciones: se recorre el índice de tiempo para el cual se tiene la ventana de entrada
    forecast_dict = {}
    start_pred = pd.Timestamp("2025-01-08 00:00:00")
    for current_time in sorted(df_norm.index):
        if current_time < start_pred or current_time > end_pred:
            continue
        window_start = current_time - pd.Timedelta(hours=INPUT_WINDOW)
        window_data = df_norm.loc[window_start: current_time - pd.Timedelta(hours=1), input_features].values
        if window_data.shape[0] != INPUT_WINDOW:
            continue
        x_input = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            # Se obtienen dos salidas: regresión y clasificación
            outputs, cls_outputs = model(x_input)
        # outputs: [OUTPUT_WINDOW, 2]; cls_outputs: [OUTPUT_WINDOW, 1]
        outputs = outputs.squeeze(0).cpu().numpy()  # [OUTPUT_WINDOW, 2]
        cls_outputs = cls_outputs.squeeze(0).cpu().numpy()  # [OUTPUT_WINDOW, 1]
        # Se combinan ambas salidas en un array de dos dimensiones, para mantener compatibilidad en la evaluación:
        # Se usará outputs para temperatura y lluvia (regresión) y cls_outputs para la clasificación.
        forecast_dict[current_time] = {'reg': outputs, 'cls': cls_outputs}
    
    # Evaluación: se extraen predicciones para horizontes 4h, 8h, 16h y 24h
    eval_start = pd.Timestamp("2025-01-11 00:00:00")
    results = []
    for current_time in sorted(forecast_dict.keys()):
        if current_time < eval_start:
            continue
        try:
            real_temp = df_norm.loc[current_time, OUTPUT_FEATURES[0]]
            real_rain = df_norm.loc[current_time, OUTPUT_FEATURES[1]]
            # Se extraen las predicciones de la rama de regresión
            reg_preds = forecast_dict[current_time]['reg']
            forecast_temp_4 = reg_preds[3][0]
            forecast_temp_8 = reg_preds[7][0]
            forecast_temp_16 = reg_preds[15][0]
            forecast_temp_24 = reg_preds[23][0]
            
            forecast_rain_4 = reg_preds[3][1]
            forecast_rain_8 = reg_preds[7][1]
            forecast_rain_16 = reg_preds[15][1]
            forecast_rain_24 = reg_preds[23][1]
            
            # Se obtienen las probabilidades de lluvia de la rama de clasificación
            # Aplicando la función sigmoide para convertir los logits en probabilidades.
            cls_preds = forecast_dict[current_time]['cls']
            prob_rain_4 = 1/(1+np.exp(-cls_preds[3][0]))
            prob_rain_8 = 1/(1+np.exp(-cls_preds[7][0]))
            prob_rain_16 = 1/(1+np.exp(-cls_preds[15][0]))
            prob_rain_24 = 1/(1+np.exp(-cls_preds[23][0]))
        except KeyError:
            continue
        
        results.append({
            'time': current_time,
            'real_temp': real_temp,
            'pred_temp_4': forecast_temp_4,
            'pred_temp_8': forecast_temp_8,
            'pred_temp_16': forecast_temp_16,
            'pred_temp_24': forecast_temp_24,
            'real_rain': real_rain,
            'pred_rain_4': forecast_rain_4,  # Probabilidad de lluvia
            'pred_rain_8': forecast_rain_8,
            'pred_rain_16': forecast_rain_16,
            'pred_rain_24': forecast_rain_24
        })
    print("Fecha mínima:", df_merged['time'].min())
    print("Fecha máxima:", df_merged['time'].max())
    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("No se generaron predicciones para el período de evaluación.")
        return
    
    # Desnormalizar las variables de regresión para temperatura y lluvia
    df_res['real_temp'] = df_res['real_temp'].apply(lambda x: x * temp_std + temp_mean)
    df_res['pred_temp_4'] = df_res['pred_temp_4'].apply(lambda x: x * temp_std + temp_mean)
    df_res['pred_temp_8'] = df_res['pred_temp_8'].apply(lambda x: x * temp_std + temp_mean)
    df_res['pred_temp_16'] = df_res['pred_temp_16'].apply(lambda x: x * temp_std + temp_mean)
    df_res['pred_temp_24'] = df_res['pred_temp_24'].apply(lambda x: x * temp_std + temp_mean)
    
    df_res['real_rain'] = df_res['real_rain'].apply(lambda x: x * rain_std + rain_mean)
    df_res['pred_rain_4'] = df_res['pred_rain_4'].apply(lambda x: x)  # Probabilidad de lluvia, no se desnormaliza
    df_res['pred_rain_8'] = df_res['pred_rain_8'].apply(lambda x: x)
    df_res['pred_rain_16'] = df_res['pred_rain_16'].apply(lambda x: x)
    df_res['pred_rain_24'] = df_res['pred_rain_24'].apply(lambda x: x)
    
    os.makedirs("results", exist_ok=True)
    
    # Gráficos para temperatura
    horizons_temp = {'4h': 'pred_temp_4', '8h': 'pred_temp_8', '16h': 'pred_temp_16', '24h': 'pred_temp_24'}
    for label, col in horizons_temp.items():
        plt.figure(figsize=(12,6))
        plt.plot(df_res['time'], df_res['real_temp'], label='Temperatura real')
        plt.plot(df_res['time'], df_res[col], label=f'Predicción {label}')
        plt.xlabel('Tiempo')
        plt.ylabel('Temperatura (°C)')
        plt.title(f'Comparación: Temperatura real vs predicción a {label}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/temperature_forecast_{label}.png")
        plt.close()
    
    # Gráficos para lluvia (usando la probabilidad de lluvia)
    horizons_rain = {'4h': 'pred_rain_4', '8h': 'pred_rain_8', '16h': 'pred_rain_16', '24h': 'pred_rain_24'}
    for label, col in horizons_rain.items():
        plt.figure(figsize=(12,6))
        plt.plot(df_res['time'], df_res['real_rain'], label='Lluvia real')
        plt.plot(df_res['time'], df_res[col], label=f'Prob. lluvia {label}')
        plt.xlabel('Tiempo')
        plt.ylabel('Probabilidad / Cantidad de lluvia')
        plt.title(f'Comparación: Lluvia real vs probabilidad de lluvia a {label}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/rain_forecast_{label}.png")
        plt.close()
    
    df_res.to_csv("results/temperature_rain_predictions_mt.csv", index=False)
    
    # -------------------------
    # Cálculo y almacenamiento de métricas de error y clasificación
    # -------------------------
    horizons = [4, 8, 16, 24]
    metrics_text = "Evaluación de Métricas:\n\n"
    print("\n=== Evaluación de Métricas ===")
    for horizon in horizons:
        metrics_text += f"Horizonte de {horizon} horas\n"
        print(f"\nHorizonte de {horizon} horas")
        # Métricas para Temperatura (regresión)
        real_temp = df_res['real_temp']
        pred_temp = df_res[f'pred_temp_{horizon}']
        mae_temp = mean_absolute_error(real_temp, pred_temp)
        mse_temp = mean_squared_error(real_temp, pred_temp)
        rmse_temp = math.sqrt(mse_temp)
        r2_temp = r2_score(real_temp, pred_temp)
        smape_temp = smape(real_temp.values, pred_temp.values)
        metrics_text += (f"Temperatura: MAE = {mae_temp:.3f}, MSE = {mse_temp:.3f}, "
                         f"RMSE = {rmse_temp:.3f}, R² = {r2_temp:.3f}, SMAPE = {smape_temp:.3f}%\n")
        print(f"Temperatura: MAE = {mae_temp:.3f}, MSE = {mse_temp:.3f}, RMSE = {rmse_temp:.3f}, R² = {r2_temp:.3f}, SMAPE = {smape_temp:.3f}%")
        
        # Métricas para Lluvia (regresión)
        # Aquí se usan las predicciones de la rama de regresión (cantidad de lluvia)
        real_rain = df_res['real_rain']
        # Para comparar, se podría evaluar la regresión de la cantidad de lluvia (aunque en este modelo la clasificación se usa para tomar decisiones)
        # Por ejemplo, se puede evaluar el MAE de la cantidad de lluvia:
        # Nota: si se desea, se puede comparar también la probabilidad vs un umbral, pero aquí se muestra la evaluación de la cantidad predicha.
        pred_rain_reg = df_res[f'pred_rain_{horizon}']  # Este valor es la probabilidad en la rama de clasificación, mientras que la rama de regresión se usó en outputs.
        # Debido a la dualidad, en esta evaluación se consideran ambas ramas:
        # Evaluamos la regresión a partir de los outputs (ya usada en desnormalización) y la clasificación con un umbral.
        # Como ejemplo, aquí usamos la cantidad (ya desnormalizada) para evaluar errores:
        mae_rain = mean_absolute_error(real_rain, pred_rain_reg)
        mse_rain = mean_squared_error(real_rain, pred_rain_reg)
        rmse_rain = math.sqrt(mse_rain)
        r2_rain = r2_score(real_rain, pred_rain_reg)
        smape_rain = smape(real_rain.values, pred_rain_reg.values)
        metrics_text += (f"Lluvia (regresión): MAE = {mae_rain:.3f}, MSE = {mse_rain:.3f}, "
                         f"RMSE = {rmse_rain:.3f}, R² = {r2_rain:.3f}, SMAPE = {smape_rain:.3f}%\n")
        print(f"Lluvia (regresión): MAE = {mae_rain:.3f}, MSE = {mse_rain:.3f}, RMSE = {rmse_rain:.3f}, R² = {r2_rain:.3f}, SMAPE = {smape_rain:.3f}%")
        
        # Métricas de clasificación para Lluvia
        # Se define lluvia como: probabilidad > 0.5
        pred_rain_binary = (df_res[f'pred_rain_{horizon}'] > 0.5).astype(int)
        real_rain_binary = (df_res['real_rain'] > 0).astype(int)
        tp = ((pred_rain_binary == 1) & (real_rain_binary == 1)).sum()
        tn = ((pred_rain_binary == 0) & (real_rain_binary == 0)).sum()
        fp = ((pred_rain_binary == 1) & (real_rain_binary == 0)).sum()
        fn = ((pred_rain_binary == 0) & (real_rain_binary == 1)).sum()
        accuracy = (tp + tn) / len(real_rain_binary)
        precision = precision_score(real_rain_binary, pred_rain_binary)
        recall = recall_score(real_rain_binary, pred_rain_binary)
        f1 = f1_score(real_rain_binary, pred_rain_binary)
        try:
            auc = roc_auc_score(real_rain_binary, df_res[f'pred_rain_{horizon}'])
        except ValueError:
            auc = float('nan')
        metrics_text += (f"Clasificación Lluvia: Accuracy = {accuracy:.3f}, Precision = {precision:.3f}, "
                         f"Recall = {recall:.3f}, F1 = {f1:.3f}, AUC-ROC = {auc:.3f} "
                         f"(TP = {tp}, TN = {tn}, FP = {fp}, FN = {fn})\n\n")
        print(f"Clasificación Lluvia: Accuracy = {accuracy:.3f}, Precision = {precision:.3f}, Recall = {recall:.3f}, F1 = {f1:.3f}, AUC-ROC = {auc:.3f} (TP = {tp}, TN = {tn}, FP = {fp}, FN = {fn})")
    
    # Escribir las métricas en un fichero de texto
    metrics_file = os.path.join("results", "metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(metrics_text)
    
    print(f"\nProceso de predicción y evaluación completado. Resultados y métricas guardados en la carpeta 'results'.")

if __name__ == '__main__':
    main()
