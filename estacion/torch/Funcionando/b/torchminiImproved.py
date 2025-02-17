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
OUTPUT_WINDOW = 12    # Predecir 72 horas siguientes
BATCH_SIZE = 32
NUM_EPOCHS = 30       # Aumentamos las épocas para mayor refinamiento
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 64
NUM_LAYERS = 2
TRAIN_RATIO = 0.8     # Fracción de datos para entrenamiento
DROPOUT = 0.2         # Dropout en LSTM

# Lista de variables meteorológicas (excluyendo columnas no numéricas o metadata)
FEATURES = ['temperature_2m (°C)', 'relative_humidity_2m (%)', 'rain (mm)',
            'pressure_msl (hPa)', 'wind_speed_10m (km/h)', 'wind_direction_10m (°)']

# Función para normalizar (y guardar parámetros para desnormalizar en predicción)
def normalize_data(df, feature_cols):
    stats = {}
    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / std
        stats[col] = (mean, std)
    return df, stats

# Dataset personalizado para series temporales
class WeatherDataset(Dataset):
    def __init__(self, df, input_window, output_window, feature_cols):
        self.df = df.reset_index(drop=True)
        self.input_window = input_window
        self.output_window = output_window
        self.feature_cols = feature_cols
        self.length = len(df) - input_window - output_window + 1
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        x = self.df.loc[idx : idx+self.input_window-1, self.feature_cols].values.astype(np.float32)
        y = self.df.loc[idx+self.input_window : idx+self.input_window+self.output_window-1, self.feature_cols].values.astype(np.float32)
        return torch.tensor(x), torch.tensor(y)

# Modelo de red neuronal: Encoder–Decoder con LSTM y Dropout
class ForecastNet(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, output_window, dropout=0.0):
        super(ForecastNet, self).__init__()
        # Si hay más de 1 capa, se activa dropout en las salidas intermedias
        self.encoder = nn.LSTM(input_size=num_features, hidden_size=hidden_size, 
                               num_layers=num_layers, batch_first=True, 
                               dropout=dropout if num_layers > 1 else 0)
        self.decoder = nn.LSTM(input_size=num_features, hidden_size=hidden_size, 
                               num_layers=num_layers, batch_first=True, 
                               dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, num_features)
        self.output_window = output_window
        
    def forward(self, src):
        # Codificación
        _, (hidden, cell) = self.encoder(src)
        # Decodificación: se utiliza el último valor del input como semilla
        decoder_input = src[:, -1:, :]  # [batch, 1, num_features]
        outputs = []
        for t in range(self.output_window):
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            pred = self.fc(out)
            outputs.append(pred)
            decoder_input = pred  # feeding autoregresivo
        outputs = torch.cat(outputs, dim=1)
        return outputs

def main():
    # Cargar datos históricos. Se salta la cabecera de metadatos si es necesario.
    data_path = 'galapagarhoraria.csv'
    df = pd.read_csv(data_path, skiprows=2, parse_dates=['time'])
    df.sort_values('time', inplace=True)
    
    # Normalización de variables y obtención de parámetros
    df_norm, stats = normalize_data(df.copy(), FEATURES)
    
    # Crear el dataset completo
    dataset = WeatherDataset(df_norm, INPUT_WINDOW, OUTPUT_WINDOW, FEATURES)
    
    # Dividir el dataset en entrenamiento y validación
    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)
    
    print(f"Número de muestras en el dataset: {len(dataset)}", flush=True)
    print(f"Número de muestras de entrenamiento: {len(train_dataset)}", flush=True)
    print(f"Número de muestras de validación: {len(val_dataset)}", flush=True)
    
    # Instanciar el modelo con dropout
    model = ForecastNet(num_features=len(FEATURES), hidden_size=HIDDEN_SIZE, 
                        num_layers=NUM_LAYERS, output_window=OUTPUT_WINDOW, dropout=DROPOUT)
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Scheduler para reducir la tasa de aprendizaje si la pérdida de validación no mejora
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Listas para almacenar las pérdidas de entrenamiento y validación
    epoch_train_losses = []
    epoch_val_losses = []
    
    # Bucle de entrenamiento
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        total_batches_train = len(train_loader)
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches_train:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Batch {batch_idx+1}/{total_batches_train} - Loss batch: {loss.item():.6f}", flush=True)
                
        train_loss /= len(train_dataset)
        epoch_train_losses.append(train_loss)
        
        # Evaluación en validación
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
    
    # Guardar modelo y parámetros de normalización
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/forecast_model.pt")
    pd.DataFrame(stats).to_csv("model/normalization_stats.csv", index=True)
    print("Modelo entrenado y parámetros guardados.", flush=True)
    
    # Graficar la evolución de la pérdida de entrenamiento y validación
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
