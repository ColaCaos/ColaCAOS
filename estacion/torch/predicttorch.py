#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de predicción para cotejar, para cada hora a partir del 11 de enero de 2025,
los valores reales (obtenidos de "galapagar25.csv") con las predicciones realizadas
respectivamente 24, 48 y 72 horas antes, usando una ventana de 168 horas (7 días).
Se usa el modelo TCNForecast entrenado (pesos en "weather_forecast_model_tcn.pt") y
el escalador guardado en "scaler.pkl". Se aprovecha la GPU si está disponible.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Parámetros del modelo y datos
INPUT_LENGTH = 168         # Ventana de 168 horas (7 días)
FORECAST_HORIZON = 72      # El modelo genera 72 horas de pronóstico
NUM_FEATURES = 5           # Variables: temperatura, humedad, lluvia, presión, velocidad del viento
HIDDEN_DIM = 64            
NUM_CHANNELS = [HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM]
KERNEL_SIZE = 3
DROPOUT = 0.2

# --- Definición de Chomp1d y TCN (igual que en el entrenamiento) ---
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x):
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size]
        else:
            return x

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
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
        out = self.net(x)
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
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                     stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

# Modelo de Forecasting basado en TCN
class TCNForecast(nn.Module):
    def __init__(self, input_length, num_features, forecast_horizon, num_channels, kernel_size, dropout):
        super(TCNForecast, self).__init__()
        # El TCN trabaja con entradas de forma (batch, channels, seq_length)
        self.tcn = TCN(num_features, num_channels, kernel_size, dropout)
        # Se toma la salida del último instante y se mapea a forecast_horizon * num_features
        self.fc = nn.Linear(num_channels[-1], forecast_horizon * num_features)
        self.forecast_horizon = forecast_horizon
        self.num_features = num_features
        
    def forward(self, x):
        # x: (batch, input_length, num_features)
        x = x.transpose(1, 2)  # (batch, num_features, input_length)
        tcn_out = self.tcn(x)  # (batch, num_channels[-1], input_length)
        last_step = tcn_out[:, :, -1]  # (batch, num_channels[-1])
        out = self.fc(last_step)       # (batch, forecast_horizon * num_features)
        out = out.view(-1, self.forecast_horizon, self.num_features)
        return out

# --- Función para cargar el archivo de datos nuevos ---
def load_new_data(csv_file):
    df = pd.read_csv(csv_file, skiprows=2)
    df["time"] = pd.to_datetime(df["time"])
    columnas = ["temperature_2m (°C)", "relative_humidity_2m (%)",
                "rain (mm)", "pressure_msl (hPa)", "wind_speed_10m (km/h)"]
    df = df[["time"] + columnas]
    for col in columnas:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return df

# --- Script principal de predicción ---
def main():
    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando dispositivo:", device)
    
    # Cargar el escalador
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    # Inicializar el modelo TCNForecast y cargar pesos
    model = TCNForecast(INPUT_LENGTH, NUM_FEATURES, FORECAST_HORIZON, NUM_CHANNELS, KERNEL_SIZE, DROPOUT).to(device)
    model.load_state_dict(torch.load("weather_forecast_model_tcn.pt", map_location=device))
    model.eval()
    
    # Cargar datos nuevos (galapagarhoraria25.csv)
    df = load_new_data("galapagarhoraria25.csv")
    df = df.sort_values("time").reset_index(drop=True)
    
    # Escalar las variables (excluyendo "time")
    data_values = df.drop("time", axis=1).values
    data_scaled = scaler.transform(data_values)
    
    # Suponiendo que los datos tienen frecuencia horaria consecutiva,
    # buscamos el primer índice correspondiente al 11 de enero de 2025 a las 00:00
    start_time = pd.Timestamp("2025-01-11 00:00:00")
    df_target = df[df["time"] >= start_time].reset_index(drop=True)
    # Además, determinamos el índice en el dataframe original (suponiendo que el primer registro es, por ejemplo, el 1 de enero)
    # Para mayor simplicidad, usaremos índices basados en la posición asumiendo que los datos son consecutivos.
    # Se requiere que para la predicción a 72h (usando la ventana del forecast issuance a T-72) se tenga al menos INPUT_LENGTH registros.
    # Entonces, para cada hora T (a partir de 11/01/2025 00:00), usaremos:
    #   - Para predicción a 24h: la ventana de [i_24 - INPUT_LENGTH, i_24) donde i_24 = índice de T - 24
    #   - Para 48h: ventana de [i_48 - INPUT_LENGTH, i_48) con i_48 = índice de T - 48
    #   - Para 72h: ventana de [i_72 - INPUT_LENGTH, i_72) con i_72 = índice de T - 72
    # Suponemos que los índices se corresponden con las horas.
    
    resultados = []
    # Determinamos el índice de inicio: T = 11/01/2025 00:00 y, para calcular T-72, necesitamos que (i - 72) >= INPUT_LENGTH
    # Así, iteramos desde i = max(start_index, INPUT_LENGTH+72) hasta el final.
    # Como los datos son consecutivos, obtenemos el índice del primer registro >= start_time:
    start_idx = df.index[df["time"] >= start_time][0]
    
    for i in range(start_idx, len(df)):
        # Verificar que se tengan suficientes datos históricos para 72h de atraso
        if i - 72 < INPUT_LENGTH:
            continue
        
        T = df["time"].iloc[i]  # Tiempo real (target) para el que se cotejará la predicción
        
        # Los índices para las predicciones:
        i24 = i - 24   # Forecast issuance para predicción a 24h (real T fue pronosticado 24h antes)
        i48 = i - 48
        i72 = i - 72
        
        # Extraer ventanas para cada forecast (la ventana tiene longitud INPUT_LENGTH)
        window24 = data_scaled[i24 - INPUT_LENGTH : i24]
        window48 = data_scaled[i48 - INPUT_LENGTH : i48]
        window72 = data_scaled[i72 - INPUT_LENGTH : i72]
        
        # Convertir a tensores y agregar dimensión batch
        input24 = torch.tensor(window24, dtype=torch.float32).unsqueeze(0).to(device)
        input48 = torch.tensor(window48, dtype=torch.float32).unsqueeze(0).to(device)
        input72 = torch.tensor(window72, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # El modelo genera una secuencia de 72 horas
            pred_seq24 = model(input24)  # (1, 72, NUM_FEATURES)
            pred_seq48 = model(input48)
            pred_seq72 = model(input72)
        
        # Extraer la predicción para el instante de interés:
        # Se extrae el valor en la posición correspondiente: para predicción a 24h, se usa índice 23 (0-indexed)
        # Para 48h, índice 47 y para 72h, índice 71.
        pred24 = pred_seq24.squeeze(0)[23].cpu().numpy()
        pred48 = pred_seq48.squeeze(0)[47].cpu().numpy()
        pred72 = pred_seq72.squeeze(0)[71].cpu().numpy()
        
        # Valor real para T (extraído de df, sin escalado)
        real = df.drop("time", axis=1).iloc[i].values
        
        # Guardar resultados: también se guarda el tiempo T
        resultados.append({
            "time": T,
            "temp_real": real[0],
            "hum_real": real[1],
            "rain_real": real[2],
            "pres_real": real[3],
            "wind_real": real[4],
            "temp_pred_24": pred24[0],
            "hum_pred_24": pred24[1],
            "rain_pred_24": pred24[2],
            "pres_pred_24": pred24[3],
            "wind_pred_24": pred24[4],
            "temp_pred_48": pred48[0],
            "hum_pred_48": pred48[1],
            "rain_pred_48": pred48[2],
            "pres_pred_48": pred48[3],
            "wind_pred_48": pred48[4],
            "temp_pred_72": pred72[0],
            "hum_pred_72": pred72[1],
            "rain_pred_72": pred72[2],
            "pres_pred_72": pred72[3],
            "wind_pred_72": pred72[4],
        })
    
    # Convertir resultados a DataFrame
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv("predicciones_comparacion_tcn.csv", index=False)
    print("Archivo CSV guardado: predicciones_comparacion_tcn.csv")
    
    # --- Visualización ---
    # Para cada parámetro, graficar en la misma figura el valor real y las predicciones a 24, 48 y 72 horas.
    parametros = {
        "Temperatura (°C)": ("temp_real", "temp_pred_24", "temp_pred_48", "temp_pred_72"),
        "Humedad (%)": ("hum_real", "hum_pred_24", "hum_pred_48", "hum_pred_72"),
        "Lluvia (mm/h)": ("rain_real", "rain_pred_24", "rain_pred_48", "rain_pred_72"),
        "Presión (hPa)": ("pres_real", "pres_pred_24", "pres_pred_48", "pres_pred_72"),
        "Velocidad (km/h)": ("wind_real", "wind_pred_24", "wind_pred_48", "wind_pred_72")
    }
    
    for param, cols in parametros.items():
        plt.figure(figsize=(10,6))
        plt.plot(df_resultados["time"], df_resultados[cols[0]], label="Real", marker="o", linestyle="-")
        plt.plot(df_resultados["time"], df_resultados[cols[1]], label="Pred 24h", marker="s", linestyle="--")
        plt.plot(df_resultados["time"], df_resultados[cols[2]], label="Pred 48h", marker="^", linestyle="--")
        plt.plot(df_resultados["time"], df_resultados[cols[3]], label="Pred 72h", marker="d", linestyle="--")
        plt.xlabel("Tiempo")
        plt.ylabel(param)
        plt.title(f"{param}: Real vs Predicciones (24h, 48h, 72h)")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"prediccion_{param.replace(' ','_').replace('(','').replace(')','')}.png")
        plt.show()

if __name__ == "__main__":
    main()
