#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de entrenamiento para la predicción meteorológica usando PyTorch con GPU en Windows 11.
Se implementa un modelo encoder–decoder con LSTM, atención y mejoras adicionales para una mejor convergencia:
- 2 capas LSTM en encoder y decoder con dropout (0.3).
- Dropout adicional (0.4) antes de la capa final.
- Optimización con weight decay (1e-3) y un learning rate reducido (1e-4).
- Scheduler con mayor paciencia (5 épocas) y early stopping aumentado a 15 épocas sin mejora.
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

# Parámetros del modelo y de la serie
INPUT_LENGTH = 168      # Ventana de 7 días (168 horas)
OUTPUT_LENGTH = 72      # Predicción de 72 horas
NUM_FEATURES = 5        # Variables: temperatura, humedad, lluvia, presión y velocidad del viento
HIDDEN_DIM = 64         # Dimensión del espacio latente
BATCH_SIZE = 32
EPOCHS = 100            # Se aumenta el número máximo de épocas
LEARNING_RATE = 1e-4    # Learning rate reducido
WEIGHT_DECAY = 1e-3     # Regularización L2

# Parámetros para early stopping
N_EPOCHS_STOP = 15      # Se incrementa la paciencia a 15 épocas sin mejora

# 1. Preprocesamiento: Cargar datos
def load_data(csv_file):
    """
    Lee el archivo CSV saltando las dos primeras líneas de metadatos,
    parsea la columna 'time' y selecciona las variables de interés.
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

# 2. Creación de secuencias (ventana de entrada y salida)
def create_sequences(data, input_len=INPUT_LENGTH, output_len=OUTPUT_LENGTH):
    """
    Genera pares de secuencias (X, y) a partir de los datos escalados.
    X: (n_muestras, input_len, NUM_FEATURES)
    y: (n_muestras, output_len, NUM_FEATURES)
    """
    X, y = [], []
    for i in range(len(data) - input_len - output_len + 1):
        X.append(data[i : i + input_len])
        y.append(data[i + input_len : i + input_len + output_len])
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

# 3. Definición del mecanismo de atención
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
    
    def forward(self, decoder_outputs, encoder_outputs):
        """
        decoder_outputs: (batch, T_dec, hidden_dim)
        encoder_outputs: (batch, T_enc, hidden_dim)
        Devuelve: vector de contexto (batch, T_dec, hidden_dim)
        """
        dec_transformed = self.W(decoder_outputs)              # (batch, T_dec, hidden_dim)
        dec_expanded = dec_transformed.unsqueeze(2)            # (batch, T_dec, 1, hidden_dim)
        enc_expanded = encoder_outputs.unsqueeze(1)            # (batch, 1, T_enc, hidden_dim)
        score = torch.tanh(dec_expanded + enc_expanded)          # (batch, T_dec, T_enc, hidden_dim)
        score = self.V(score).squeeze(-1)                        # (batch, T_dec, T_enc)
        attention_weights = torch.softmax(score, dim=-1)         # (batch, T_dec, T_enc)
        context = torch.bmm(attention_weights, encoder_outputs)  # (batch, T_dec, hidden_dim)
        return context

# 4. Definición del modelo encoder–decoder con atención, 2 capas LSTM y dropout adicional
class Seq2SeqAttention(nn.Module):
    def __init__(self, input_len, output_len, num_features, hidden_dim):
        super(Seq2SeqAttention, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        
        # Encoder: 2 capas LSTM con dropout (0.3)
        self.encoder = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        # Decoder: 2 capas LSTM con dropout (0.3)
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        # Mecanismo de atención
        self.attention = Attention(hidden_dim)
        # Dropout adicional antes de la capa final (0.4)
        self.dropout = nn.Dropout(0.4)
        # Capa final para predecir las variables
        self.fc = nn.Linear(2 * hidden_dim, num_features)
    
    def forward(self, x):
        # x: (batch, input_len, num_features)
        encoder_outputs, (hidden, cell) = self.encoder(x)   # encoder_outputs: (batch, input_len, hidden_dim)
        # Para el decoder, usamos el último estado de la última capa y lo repetimos para cada paso de salida
        dec_input = hidden[-1].unsqueeze(1).repeat(1, self.output_len, 1)  # (batch, output_len, hidden_dim)
        decoder_outputs, _ = self.decoder(dec_input, (hidden, cell))         # (batch, output_len, hidden_dim)
        context = self.attention(decoder_outputs, encoder_outputs)           # (batch, output_len, hidden_dim)
        combined = torch.cat((decoder_outputs, context), dim=2)              # (batch, output_len, 2*hidden_dim)
        combined = self.dropout(combined)                                    # Aplicar dropout adicional
        output = self.fc(combined)                                           # (batch, output_len, num_features)
        return output

# 5. Proceso principal de entrenamiento con scheduler y early stopping
def main():
    # Definir el dispositivo: GPU si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando dispositivo:", device)
    
    # Cargar datos históricos (por ejemplo, "galapagarhoraria.csv")
    df = load_data('galapagarhoraria.csv')
    print("Datos cargados:", df.shape)
    
    # Seleccionar las variables (excluyendo 'time')
    data_values = df.drop('time', axis=1).values

    # Escalar los datos
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_values)
    
    # Guardar el escalador para uso en predicción
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Crear secuencias para entrenamiento
    X, y = create_sequences(data_scaled)
    print("Forma de X:", X.shape)
    print("Forma de y:", y.shape)
    
    # División en entrenamiento y validación (80%/20% de forma secuencial)
    split_index = int(0.8 * len(X))
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]
    
    # Crear datasets y dataloaders
    train_dataset = WeatherDataset(X_train, y_train)
    val_dataset = WeatherDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Inicializar el modelo, la función de pérdida y el optimizador (con weight decay)
    model = Seq2SeqAttention(INPUT_LENGTH, OUTPUT_LENGTH, NUM_FEATURES, HIDDEN_DIM).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Scheduler para reducir el learning rate si no hay mejora en la pérdida de validación
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)
    
    best_val_loss = np.inf
    epochs_no_improve = 0  # Contador para early stopping
    
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
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validación
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # Actualizar el scheduler
        scheduler.step(epoch_val_loss)
        
        print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {epoch_train_loss:.6f} - Val Loss: {epoch_val_loss:.6f}")
        
        # Early stopping: reiniciar el contador si hay mejora, de lo contrario incrementarlo
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'weather_forecast_model.pt')
            print("  --> Modelo guardado.")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= N_EPOCHS_STOP:
            print(f"Early stopping: no mejora en la pérdida de validación durante {N_EPOCHS_STOP} épocas consecutivas.")
            break
    
    # Graficar la evolución de la pérdida
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.title("Pérdida durante el entrenamiento")
    plt.legend()
    plt.savefig("training_loss.png")
    plt.show()

if __name__ == '__main__':
    main()
