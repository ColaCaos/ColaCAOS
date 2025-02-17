#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de entrenamiento para la predicción meteorológica usando un modelo TCN en PyTorch.
El modelo toma una ventana de 168 horas de datos (5 variables) y predice 72 horas en el futuro.
Se utiliza GPU (si está disponible), se aplica early stopping y se guarda el mejor modelo.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Parámetros
INPUT_LENGTH = 168         # Horas de historia (7 días)
FORECAST_HORIZON = 72      # Horas a predecir
NUM_FEATURES = 5           # Variables: temperatura, humedad, lluvia, presión y velocidad del viento
HIDDEN_DIM = 64            # Dimensión interna en el TCN
NUM_CHANNELS = [HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM]  # Canales para cada capa TCN
KERNEL_SIZE = 3            # Tamaño del kernel de las convoluciones
DROPOUT = 0.2              # Dropout en las capas TCN
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3

# Parámetros para early stopping
PATIENCE = 15

# --- Definición de Chomp1d ---
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x):
        # x tiene forma (batch, channels, length)
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size]
        else:
            return x

# --- Definición de la arquitectura TCN ---
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)  # Recorta la salida para que tenga longitud original
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, kernel_size=1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.net(x)  # La salida tendrá ahora la misma longitud que x
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size  # padding para mantener la longitud, se recorta luego
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                     stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

# Modelo de Forecasting basado en TCN
class TCNForecast(nn.Module):
    def __init__(self, input_length, num_features, forecast_horizon, num_channels, kernel_size, dropout):
        """
        input_length: longitud de la secuencia de entrada (168)
        num_features: número de variables (5)
        forecast_horizon: número de pasos a predecir (72)
        num_channels: lista con la cantidad de filtros en cada capa TCN
        """
        super(TCNForecast, self).__init__()
        # El TCN espera la entrada con forma (batch, channels, seq_length)
        self.tcn = TCN(num_features, num_channels, kernel_size, dropout)
        # Se toma la salida en el último instante y se mapea a forecast_horizon * num_features
        self.fc = nn.Linear(num_channels[-1], forecast_horizon * num_features)
        self.forecast_horizon = forecast_horizon
        self.num_features = num_features
        
    def forward(self, x):
        # x: (batch, input_length, num_features)
        x = x.transpose(1, 2)  # Convertir a (batch, num_features, input_length)
        tcn_out = self.tcn(x)  # (batch, num_channels[-1], input_length) – la longitud se conserva
        last_step = tcn_out[:, :, -1]  # Tomamos el último instante: (batch, num_channels[-1])
        out = self.fc(last_step)       # (batch, forecast_horizon * num_features)
        out = out.view(-1, self.forecast_horizon, self.num_features)
        return out

# --- Preprocesamiento de datos y creación de secuencias ---
def load_data(csv_file):
    """
    Carga el CSV saltando las dos primeras líneas de metadatos, convierte 'time' a datetime,
    y selecciona las variables de interés.
    """
    df = pd.read_csv(csv_file, skiprows=2)
    df['time'] = pd.to_datetime(df['time'])
    columnas = ["temperature_2m (°C)", "relative_humidity_2m (%)", 
                "rain (mm)", "pressure_msl (hPa)", "wind_speed_10m (km/h)"]
    df = df[['time'] + columnas]
    for col in columnas:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    return df

def create_sequences(data, input_len=INPUT_LENGTH, forecast_horizon=FORECAST_HORIZON):
    """
    Genera pares (X, y) donde:
      X: secuencias de entrada con forma (n_samples, input_len, NUM_FEATURES)
      y: secuencias de salida con forma (n_samples, forecast_horizon, NUM_FEATURES)
    """
    X, y = [], []
    total_length = input_len + forecast_horizon
    for i in range(len(data) - total_length + 1):
        X.append(data[i : i + input_len])
        y.append(data[i + input_len : i + total_length])
    return np.array(X), np.array(y)

# Dataset personalizado
class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- Script principal de entrenamiento ---
def main():
    # Definir el dispositivo (GPU si está disponible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando dispositivo:", device)
    
    # Cargar datos históricos (por ejemplo, "galapagarhoraria.csv")
    df = load_data("galapagarhoraria.csv")
    print("Datos cargados:", df.shape)
    
    # Seleccionar las variables (excluyendo la columna 'time')
    data_values = df.drop("time", axis=1).values
    
    # Escalar los datos
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_values)
    
    # Guardar el escalador para uso en predicción
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    # Crear secuencias para entrenamiento y validación
    X, y = create_sequences(data_scaled)
    print("Forma de X:", X.shape)
    print("Forma de y:", y.shape)
    
    # División en entrenamiento y validación (80% / 20%)
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    train_dataset = WeatherDataset(X_train, y_train)
    val_dataset = WeatherDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Inicializar el modelo TCNForecast
    model = TCNForecast(INPUT_LENGTH, NUM_FEATURES, FORECAST_HORIZON, NUM_CHANNELS, KERNEL_SIZE, DROPOUT).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5, verbose=True)
    
    best_val_loss = float("inf")
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_train_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)
        
        # Evaluación en el conjunto de validación
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)
        epoch_val_loss = running_val_loss / len(val_dataset)
        val_losses.append(epoch_val_loss)
        
        scheduler.step(epoch_val_loss)
        
        print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {epoch_train_loss:.6f} - Val Loss: {epoch_val_loss:.6f}")
        
        # Guarda el mejor modelo y controla early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "weather_forecast_model_tcn.pt")
            print("  --> Modelo guardado.")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping: no mejora en la pérdida de validación durante {PATIENCE} épocas consecutivas.")
            break
    
    # Graficar la evolución de la pérdida
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.title("Evolución de la pérdida durante el entrenamiento")
    plt.legend()
    plt.savefig("training_loss_tcn.png")
    plt.show()

if __name__ == "__main__":
    main()
