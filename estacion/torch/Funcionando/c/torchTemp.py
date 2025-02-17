import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import os
import matplotlib.pyplot as plt

# Configuración de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo utilizado:", device, flush=True)

# Hiperparámetros
INPUT_WINDOW = 168    # 168 horas = 7 días de datos pasados
OUTPUT_WINDOW = 12    # Predecir 12 horas siguientes
BATCH_SIZE = 32
NUM_EPOCHS = 30       
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 64
NUM_LAYERS = 2
TRAIN_RATIO = 0.8     # Fracción de datos para entrenamiento
DROPOUT = 0.2         # Dropout en LSTM

# Lista de variables de entrada (features) – se usan todas para el input
FEATURES = ['temperature_2m (°C)', 'relative_humidity_2m (%)', 'rain (mm)',
            'pressure_msl (hPa)', 'wind_speed_10m (km/h)', 'wind_direction_10m (°)']
# Variable objetivo (output): solo temperatura
OUTPUT_FEATURE = 'temperature_2m (°C)'

# Función para normalizar (se normalizan todas las variables)
def normalize_data(df, feature_cols):
    stats = {}
    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / std
        stats[col] = (mean, std)
    return df, stats

# Dataset personalizado para series temporales:
# La entrada son todas las features y la salida es únicamente la temperatura.
class WeatherDataset(Dataset):
    def __init__(self, df, input_window, output_window, input_features, output_feature):
        self.df = df.reset_index(drop=True)
        self.input_window = input_window
        self.output_window = output_window
        self.input_features = input_features
        self.output_feature = output_feature
        self.length = len(df) - input_window - output_window + 1
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Input: todas las features en la ventana de entrada
        x = self.df.loc[idx : idx+self.input_window-1, self.input_features].values.astype(np.float32)
        # Output: solo la temperatura durante la ventana de salida, con forma [output_window, 1]
        y = self.df.loc[idx+self.input_window : idx+self.input_window+self.output_window-1, self.output_feature].values.astype(np.float32)
        y = y.reshape(-1, 1)
        return torch.tensor(x), torch.tensor(y)

# Modelo: Encoder–Decoder con LSTM, Dropout y proyección para ajustar la dimensión del decoder input.
class ForecastNet(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, output_window, dropout=0.0):
        super(ForecastNet, self).__init__()
        # Encoder: usa todas las features.
        self.encoder = nn.LSTM(input_size=num_features, hidden_size=hidden_size, 
                               num_layers=num_layers, batch_first=True, 
                               dropout=dropout if num_layers > 1 else 0)
        # Decoder: se espera input de dimensión num_features.
        self.decoder = nn.LSTM(input_size=num_features, hidden_size=hidden_size, 
                               num_layers=num_layers, batch_first=True, 
                               dropout=dropout if num_layers > 1 else 0)
        # Capa final para producir un valor (temperatura).
        self.fc = nn.Linear(hidden_size, 1)
        self.output_window = output_window
        # Proyección para transformar un valor (1-dim) a un vector de tamaño num_features,
        # de modo que siempre se alimente el decoder con la dimensión esperada.
        self.proj = nn.Linear(1, num_features)
        
    def forward(self, src):
        # src: [batch, INPUT_WINDOW, num_features]
        _, (hidden, cell) = self.encoder(src)
        # Inicial: usar el último valor de temperatura de la secuencia de entrada.
        # Suponemos que la temperatura es la primera columna de FEATURES.
        decoder_input = src[:, -1:, 0:1]  # [batch, 1, 1]
        outputs = []
        for t in range(self.output_window):
            # Proyectar el valor (1-dim) a un vector de dimensión num_features.
            decoder_input_proj = self.proj(decoder_input)  # [batch, 1, num_features]
            out, (hidden, cell) = self.decoder(decoder_input_proj, (hidden, cell))
            pred = self.fc(out)  # [batch, 1, 1] -> predicción de temperatura para este paso
            outputs.append(pred)
            # Para el siguiente paso, usamos la predicción actual (manteniéndola en 1 dimensión)
            decoder_input = pred
        outputs = torch.cat(outputs, dim=1)  # [batch, output_window, 1]
        return outputs

def main():
    # Cargar datos históricos. Se salta la cabecera de metadatos si es necesario.
    data_path = 'galapagarhoraria.csv'
    df = pd.read_csv(data_path, skiprows=2, parse_dates=['time'])
    df.sort_values('time', inplace=True)
    
    # Normalización de todas las variables (input). Se guardan las estadísticas para todas.
    df_norm, stats = normalize_data(df.copy(), FEATURES)
    
    # Crear el dataset (input: FEATURES; target: OUTPUT_FEATURE)
    dataset = WeatherDataset(df_norm, INPUT_WINDOW, OUTPUT_WINDOW, FEATURES, OUTPUT_FEATURE)
    
    # Dividir en entrenamiento y validación
    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)
    
    print(f"Número de muestras en el dataset: {len(dataset)}", flush=True)
    print(f"Número de muestras de entrenamiento: {len(train_dataset)}", flush=True)
    print(f"Número de muestras de validación: {len(val_dataset)}", flush=True)
    
    # Instanciar el modelo (input_size = len(FEATURES), output = 1)
    model = ForecastNet(num_features=len(FEATURES), hidden_size=HIDDEN_SIZE, 
                        num_layers=NUM_LAYERS, output_window=OUTPUT_WINDOW, dropout=DROPOUT)
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    epoch_train_losses = []
    epoch_val_losses = []
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        total_batches_train = len(train_loader)
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)  # [batch, OUTPUT_WINDOW, 1]
            optimizer.zero_grad()
            output = model(batch_x)  # [batch, OUTPUT_WINDOW, 1]
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches_train:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Batch {batch_idx+1}/{total_batches_train} - Loss batch: {loss.item():.6f}", flush=True)
                
        train_loss /= len(train_dataset)
        epoch_train_losses.append(train_loss)
        
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
        epoch_val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss entrenamiento: {train_loss:.6f} - Loss validación: {val_loss:.6f}", flush=True)
        scheduler.step(val_loss)
    
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/forecast_model.pt")
    pd.DataFrame(stats).to_csv("model/normalization_stats.csv", index=True)
    print("Modelo entrenado y parámetros guardados.", flush=True)
    
    plt.figure(figsize=(8,5))
    plt.plot(range(1, NUM_EPOCHS+1), epoch_train_losses, marker='o', label="Entrenamiento")
    plt.plot(range(1, NUM_EPOCHS+1), epoch_val_losses, marker='o', label="Validación")
    plt.xlabel("Epoch")
    plt.ylabel("Pérdida (MSE)")
    plt.title("Evolución de la pérdida durante el entrenamiento")
    plt.legend()
    plt.grid(True)
    plt.savefig("model/training_validation_loss.png")
    plt.show()

if __name__ == '__main__':
    main()
