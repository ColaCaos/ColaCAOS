#!/usr/bin/env python
"""
Training script for predicting rain probability using only barometric pressure and humidity.
This script loads data from 'galapagarhoraria.csv', computes the binary rain probability target,
normalizes the input features, creates a sliding‐window dataset, trains a simplified LSTM encoder–decoder,
and saves the trained model along with normalization statistics.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Hyperparameters and data parameters
INPUT_WINDOW = 168    # Input sequence length (hours, e.g., 7 days)
OUTPUT_WINDOW = 12    # Forecast horizon (hours)
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
TRAIN_RATIO = 0.8

# Define the input features (only pressure and humidity) and target
INPUT_FEATURES = ['pressure_msl (hPa)', 'relative_humidity_2m (%)']
TARGET_FEATURE = 'rain_probability'

# Function to create binary rain probability target (1 if rain > 0, else 0)
def create_rain_probability(df):
    df[TARGET_FEATURE] = (df['rain (mm)'] > 0).astype(float)
    return df

# Normalize specified columns and save the computed statistics
def normalize_data(df, cols):
    stats = {}
    df_norm = df.copy()
    for col in cols:
        mean = df_norm[col].mean()
        std = df_norm[col].std()
        df_norm[col] = (df_norm[col] - mean) / std
        stats[col] = {'mean': mean, 'std': std}
    return df_norm, stats

# Custom Dataset for sliding-window sequences
class WeatherDataset(Dataset):
    def __init__(self, df, input_window, output_window, input_cols, target_col):
        self.df = df.reset_index(drop=True)
        self.input_window = input_window
        self.output_window = output_window
        self.input_cols = input_cols
        self.target_col = target_col
        self.length = len(self.df) - input_window - output_window + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.df.loc[idx: idx+self.input_window-1, self.input_cols].values.astype(np.float32)
        y = self.df.loc[idx+self.input_window: idx+self.input_window+self.output_window-1, self.target_col].values.astype(np.float32)
        y = y.reshape(-1, 1)
        return torch.tensor(x), torch.tensor(y)

# Simplified LSTM encoder-decoder model using only two input features
class ForecastNetSimple(nn.Module):
    def __init__(self, input_size, hidden_size, output_window, dropout=0.0):
        super(ForecastNetSimple, self).__init__()
        self.hidden_size = hidden_size
        self.output_window = output_window
        # Encoder LSTM: input_size matches number of input features (2)
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=1, batch_first=True)
        # Decoder LSTM: we use the same LSTM cell for decoding
        self.decoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=1, batch_first=True)
        # Projection to convert decoder input (scalar) to input_size dimension
        self.proj = nn.Linear(1, input_size)
        # Final fully-connected layer to output a single probability value
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, src, target_seq=None, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        # Encode the input sequence
        _, (hidden, cell) = self.encoder(src)
        # Initialize decoder input: use last pressure value (arbitrary choice)
        decoder_input = src[:, -1:, 0:1]  # Shape: (batch, 1, 1)
        outputs = []
        for t in range(self.output_window):
            # Project decoder input to match the input feature dimension
            decoder_input_proj = self.proj(decoder_input)
            out, (hidden, cell) = self.decoder(decoder_input_proj, (hidden, cell))
            pred = torch.sigmoid(self.fc(out))
            outputs.append(pred)
            # Optionally use teacher forcing
            if target_seq is not None and random.random() < teacher_forcing_ratio:
                decoder_input = target_seq[:, t:t+1, :]
            else:
                decoder_input = pred
        outputs = torch.cat(outputs, dim=1)  # Shape: (batch, output_window, 1)
        return outputs

def main():
    # Load and sort the training data
    data_path = 'galapagarhoraria.csv'
    df = pd.read_csv(data_path, skiprows=2, parse_dates=['time'])
    df.sort_values('time', inplace=True)

    # Create binary rain probability target
    df = create_rain_probability(df)

    # Normalize only the selected input features (do not normalize the binary target)
    df_norm, norm_stats = normalize_data(df, INPUT_FEATURES)
    os.makedirs("model", exist_ok=True)
    pd.DataFrame(norm_stats).to_csv("model/normalization_stats_simple.csv")

    # Create the dataset and split into training and validation sets
    dataset = WeatherDataset(df_norm, INPUT_WINDOW, OUTPUT_WINDOW, INPUT_FEATURES, TARGET_FEATURE)
    train_size = int(len(dataset) * TRAIN_RATIO)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize the model
    model = ForecastNetSimple(input_size=len(INPUT_FEATURES), hidden_size=64, output_window=OUTPUT_WINDOW, dropout=0.0)
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x, target_seq=batch_y, teacher_forcing_ratio=0.5)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * batch_x.size(0)
        epoch_train_loss /= len(train_dataset)
        train_losses.append(epoch_train_loss)
        
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                output = model(batch_x)
                loss = criterion(output, batch_y)
                epoch_val_loss += loss.item() * batch_x.size(0)
        epoch_val_loss /= len(val_dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}", flush=True)

    # Save the trained model
    torch.save(model.state_dict(), "model/forecast_model_simple.pt")
    
    # Plot and save the training and validation loss curve
    plt.figure(figsize=(10,6))
    plt.plot(range(1, NUM_EPOCHS+1), train_losses, label="Training Loss")
    plt.plot(range(1, NUM_EPOCHS+1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("model/training_validation_loss_simple.png")
    plt.show()

if __name__ == '__main__':
    main()
