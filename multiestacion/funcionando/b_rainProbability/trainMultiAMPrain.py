import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import random
from functools import reduce
from multiprocessing import freeze_support

# -------------------------
# Función de normalización (definida globalmente)
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
    df = pd.read_csv(filename, skiprows=2, parse_dates=['time'])
    df.sort_values('time', inplace=True)
    df = add_temporal_features(df, time_col='time')
    if station != 'galapagar':
        rename_dict = {col: f"{station}_{col}" for col in BASE_FEATURES}
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

# Target original: se predecía temperatura y lluvia.
# Ahora, para la lluvia usaremos dos ramas: una para la regresión y otra para la clasificación.
TARGET_FEATURES = ['temperature_2m (°C)', 'rain (mm)']

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
NUM_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
TEACHER_FORCING_RATIO = 0.5

# Pesos para cada término de la pérdida
ALPHA = 1.0   # Pérdida para la regresión de lluvia
BETA  = 2.0   # Pérdida para la clasificación de lluvia

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
    'guadalajara': 'guadalajara.csv',
    'segovia': 'segovia.csv',
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
        # Los targets serán: temperatura y lluvia (regresión).
        y = self.df.loc[idx + self.input_window : idx + self.input_window + self.output_window - 1, self.target_features].values.astype(np.float32)
        return torch.tensor(x), torch.tensor(y)

# -------------------------
# Modelo Multi-Tarea: ForecastNetMT
# -------------------------
class ForecastNetMT(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, output_window, output_dim, dropout=0.0):
        """
        Args:
            num_features: Número total de features de entrada.
            hidden_size: Tamaño de la capa oculta.
            num_layers: Número de capas en el LSTM.
            output_window: Número de pasos de tiempo a predecir.
            output_dim: Dimensión de la salida para los targets principales (temperatura y lluvia regresión).
            dropout: Tasa de dropout.
        """
        super(ForecastNetMT, self).__init__()
        self.encoder = nn.LSTM(input_size=num_features, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        # El decoder usará el mismo esquema de retroalimentación.
        self.decoder = nn.LSTM(input_size=num_features, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        # Capa final para los targets principales: [temperatura, lluvia_regresión]
        self.fc = nn.Linear(hidden_size, output_dim)
        # Rama auxiliar para la clasificación de lluvia (salida logit, 1 valor por paso de tiempo)
        self.cls_fc = nn.Linear(hidden_size, 1)
        self.proj = nn.Linear(output_dim, num_features)
        self.output_window = output_window

    def forward(self, src, target_seq=None, teacher_forcing_ratio=0.0):
        # Codificación
        _, (hidden, cell) = self.encoder(src)
        # Inicializar el decoder. Se usan las variables objetivo de Galapagar.
        # Se asume que en el DataFrame, para Galapagar, la temperatura está en la posición 0 y la lluvia en la posición 2.
        decoder_input = src[:, -1:, :][:, :, [0, 2]]  # [batch, 1, 2]
        outputs = []       # Salida principal: [temperatura, lluvia_regresión]
        cls_outputs = []   # Salida auxiliar: clasificación de lluvia (logits)
        for t in range(self.output_window):
            decoder_input_proj = self.proj(decoder_input)  # [batch, 1, num_features]
            out, (hidden, cell) = self.decoder(decoder_input_proj, (hidden, cell))
            pred = self.fc(out)  # [batch, 1, 2]
            # Rama auxiliar: clasificación de lluvia (a partir del estado oculto)
            cls_logit = self.cls_fc(out)  # [batch, 1, 1]
            outputs.append(pred)
            cls_outputs.append(cls_logit)
            # Teacher forcing: se utiliza la secuencia objetivo en vez de la propia predicción.
            if target_seq is not None and np.random.rand() < teacher_forcing_ratio:
                # Se alimenta el target (para ambas variables) al decoder.
                decoder_input = target_seq[:, t:t+1, :]
            else:
                decoder_input = pred
        outputs = torch.cat(outputs, dim=1)       # [batch, output_window, 2]
        cls_outputs = torch.cat(cls_outputs, dim=1)   # [batch, output_window, 1]
        return outputs, cls_outputs

# -------------------------
# Función principal de entrenamiento
# -------------------------
def main():
    # Cargar datos de cada estación y fusionarlos
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
    
    input_features = []
    for station in station_files.keys():
        if station == 'galapagar':
            input_features += BASE_FEATURES
        else:
            input_features += [f"{station}_{col}" for col in BASE_FEATURES]
    target_features = TARGET_FEATURES  # [temperatura, lluvia] (esta última se usa para la rama de regresión)
    
    # Normalizar los datos
    df_norm, norm_stats = normalize_data(df_merged, input_features)
    os.makedirs("model", exist_ok=True)
    stats_df = pd.DataFrame({k: {'mean': v[0], 'std': v[1]} for k, v in norm_stats.items()}).T
    stats_df.to_csv("model/normalization_stats.csv", index=True)
    
    dataset = WeatherDataset(df_norm, INPUT_WINDOW, OUTPUT_WINDOW, input_features, target_features)
    print(f"Número total de muestras en el dataset: {len(dataset)}")
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Muestras entrenamiento: {len(train_dataset)}; validación: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    num_input_features = len(input_features)
    model = ForecastNetMT(num_features=num_input_features, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                          output_window=OUTPUT_WINDOW, output_dim=len(TARGET_FEATURES), dropout=DROPOUT)
    model = model.to(device)
    print(f"Modelo instanciado con num_features = {num_input_features}")
    
    # Definir las pérdidas:
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Inicializar GradScaler para AMP (entrenamiento de precisión mixta)
    scaler = torch.cuda.amp.GradScaler()
    
    epoch_train_losses = []
    epoch_val_losses = []
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        total_batches = len(train_loader)
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                # Salidas: 
                #   outputs: [batch, output_window, 2] (temperatura, lluvia_regresión)
                #   cls_outputs: [batch, output_window, 1] (logits para lluvia)
                outputs, cls_outputs = model(batch_x, target_seq=batch_y, teacher_forcing_ratio=TEACHER_FORCING_RATIO)
                # Separar targets de temperatura y lluvia (regresión)
                target_temp = batch_y[:, :, 0]
                target_rain = batch_y[:, :, 1]
                pred_temp = outputs[:, :, 0]
                pred_rain = outputs[:, :, 1]
                
                # Pérdida para temperatura y lluvia (regresión)
                loss_temp = mse_loss(pred_temp, target_temp)
                loss_rain_reg = mse_loss(pred_rain, target_rain)
                
                # Para la clasificación, definimos la etiqueta: 1 si target_rain > 0, 0 en caso contrario.
                target_rain_cls = (target_rain > 0).float()
                # cls_outputs tiene forma [batch, output_window, 1], se aplana para la pérdida.
                loss_rain_cls = bce_loss(cls_outputs.squeeze(-1), target_rain_cls)
                
                loss = loss_temp + ALPHA * loss_rain_reg + BETA * loss_rain_cls
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * batch_x.size(0)
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Batch {batch_idx+1}/{total_batches} - Loss: {loss.item():.6f}")
        
        train_loss /= len(train_dataset)
        epoch_train_losses.append(train_loss)
    
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                with torch.cuda.amp.autocast():
                    outputs, cls_outputs = model(batch_x)
                    target_temp = batch_y[:, :, 0]
                    target_rain = batch_y[:, :, 1]
                    pred_temp = outputs[:, :, 0]
                    pred_rain = outputs[:, :, 1]
                    
                    loss_temp = mse_loss(pred_temp, target_temp)
                    loss_rain_reg = mse_loss(pred_rain, target_rain)
                    target_rain_cls = (target_rain > 0).float()
                    loss_rain_cls = bce_loss(cls_outputs.squeeze(-1), target_rain_cls)
                    
                    loss = loss_temp + ALPHA * loss_rain_reg + BETA * loss_rain_cls
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_dataset)
        epoch_val_losses.append(val_loss)
    
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} completada - Pérdida entrenamiento: {train_loss:.6f} - Pérdida validación: {val_loss:.6f}")
        scheduler.step(val_loss)
    
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/forecast_model_mt.pt")
    print("Modelo entrenado y parámetros guardados.")
    
    plt.figure(figsize=(8,5))
    plt.plot(range(1, NUM_EPOCHS+1), epoch_train_losses, marker='o', label="Entrenamiento")
    plt.plot(range(1, NUM_EPOCHS+1), epoch_val_losses, marker='o', label="Validación")
    plt.xlabel("Epoch")
    plt.ylabel("Pérdida compuesta")
    plt.title("Evolución de la pérdida durante el entrenamiento")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("model/training_validation_loss_mt.png")
    plt.show()

if __name__ == '__main__':
    freeze_support()
    main()
