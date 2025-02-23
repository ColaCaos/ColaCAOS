import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import optuna
from functools import reduce
import matplotlib.pyplot as plt
import random

# -------------------------
# Función de normalización
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

def load_station_data(station, filename, base_features):
    df = pd.read_csv(filename, skiprows=2, parse_dates=['time'])
    df.sort_values('time', inplace=True)
    df = add_temporal_features(df, time_col='time')
    if station != 'galapagar':
        rename_dict = {col: f"{station}_{col}" for col in base_features}
        df.rename(columns=rename_dict, inplace=True)
    return df

# -------------------------
# Variables globales y configuraciones
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo utilizado:", device, flush=True)

INPUT_WINDOW = 72    # 72 horas (3 días)
OUTPUT_WINDOW = 24   # 24 horas siguientes

FEATURES = ['temperature_2m (°C)', 'relative_humidity_2m (%)', 'rain (mm)',
            'surface_pressure (hPa)', 'wind_speed_10m (km/h)', 'wind_direction_10m (°)']
TEMPORAL_FEATURES = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
BASE_FEATURES = FEATURES + TEMPORAL_FEATURES  # 10 variables por estación

TARGET_FEATURES = ['temperature_2m (°C)', 'rain (mm)']  # Se predicen solo para Galapagar

# Diccionario de archivos (asegúrate de que los ficheros existan en la ruta)
station_files = {
    'galapagar': 'galapagarhoraria.csv',
    'pontevedra': 'pontevedra.csv',
    'salamanca': 'salamanca.csv',
    'huelva': 'huelva.csv',
    'ciudadreal': 'ciudadreal.csv',
    'valencia': 'valencia.csv',
    'almeria': 'almeria.csv',
    'creus': 'creus.csv',
    'santander': 'santander.csv',
    'segovia': 'segovia.csv',
    'guadalajara': 'guadalajara.csv',
    'burgos': 'burgos.csv'
}

# -------------------------
# Dataset personalizado
# -------------------------
class WeatherDataset(Dataset):
    def __init__(self, df, input_window, output_window, input_features, target_features):
        self.df = df.reset_index(drop=True)
        self.input_window = input_window
        self.output_window = output_window
        self.input_features = input_features
        self.target_features = target_features
        self.length = len(df) - input_window - output_window + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.df.loc[idx : idx + self.input_window - 1, self.input_features].values.astype(np.float32)
        y = self.df.loc[idx + self.input_window : idx + self.input_window + self.output_window - 1, self.target_features].values.astype(np.float32)
        return torch.tensor(x), torch.tensor(y)

# -------------------------
# Modelo: ForecastNet
# -------------------------
class ForecastNet(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, output_window, output_dim, dropout=0.0):
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

    def forward(self, src, target_seq=None, teacher_forcing_ratio=0.0):
        _, (hidden, cell) = self.encoder(src)
        decoder_input = src[:, -1:, :][:, :, [0, 2]]  # Asumimos que temperatura y lluvia de Galapagar están en posición 0 y 2
        outputs = []
        for t in range(self.output_window):
            decoder_input_proj = self.proj(decoder_input)
            out, (hidden, cell) = self.decoder(decoder_input_proj, (hidden, cell))
            pred = self.fc(out)
            outputs.append(pred)
            if target_seq is not None and np.random.rand() < teacher_forcing_ratio:
                decoder_input = target_seq[:, t:t+1, :]
            else:
                decoder_input = pred
        outputs = torch.cat(outputs, dim=1)
        return outputs

# -------------------------
# Función para cargar y preprocesar los datos
# -------------------------
def load_data():
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
    input_features = []
    for station in station_files.keys():
        if station == 'galapagar':
            input_features += BASE_FEATURES
        else:
            input_features += [f"{station}_{col}" for col in BASE_FEATURES]
    df_norm, norm_stats = normalize_data(df_merged, input_features)
    df_norm.set_index('time', inplace=True)
    return df_norm, input_features

# -------------------------
# Función objetivo para Optuna
# -------------------------
def objective(trial):
    # Espacio de búsqueda para hiperparámetros
    hidden_size = trial.suggest_int("hidden_size", 32, 128)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    teacher_forcing_ratio = trial.suggest_float("teacher_forcing_ratio", 0.0, 0.8)

    # Cargar datos
    df_norm, input_features = load_data()
    dataset = WeatherDataset(df_norm, INPUT_WINDOW, OUTPUT_WINDOW, input_features, TARGET_FEATURES)
    
    # Dividir en entrenamiento y validación (80/20) respetando la secuencia
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    num_input_features = len(input_features)
    model = ForecastNet(num_features=num_input_features, hidden_size=hidden_size, num_layers=num_layers,
                        output_window=OUTPUT_WINDOW, output_dim=len(TARGET_FEATURES), dropout=dropout)
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Entrenamiento breve para la optimización (por ejemplo, 5 epochs)
    EPOCHS_OPTUNA = 5
    model.train()
    for epoch in range(EPOCHS_OPTUNA):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x, target_seq=batch_y, teacher_forcing_ratio=teacher_forcing_ratio)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        epoch_loss /= len(train_dataset)
        # Puedes imprimir la pérdida por época si lo deseas:
        # print(f"Epoch {epoch+1}/{EPOCHS_OPTUNA} - Loss: {epoch_loss:.6f}")
    
    # Evaluación en el conjunto de validación
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            output = model(batch_x)
            loss = criterion(output, batch_y)
            val_loss += loss.item() * batch_x.size(0)
    val_loss /= len(val_dataset)
    
    return val_loss

# -------------------------
# Main: Crear estudio y optimizar
# -------------------------
def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    
    print("Mejores hiperparámetros encontrados:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    print(f"Mejor MSE en validación: {study.best_value:.6f}")
    
    # Guardar resultados del estudio (opcional)
    study.trials_dataframe().to_csv("optuna_study_results.csv", index=False)

if __name__ == '__main__':
    main()
