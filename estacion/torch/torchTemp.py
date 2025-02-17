import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import os
import matplotlib.pyplot as plt
import random
import math

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device used:", device, flush=True)

# Hyperparameters
INPUT_WINDOW = 168    # 7 days (hours)
OUTPUT_WINDOW = 12    # 12 hours to predict
BATCH_SIZE = 32
NUM_EPOCHS = 30       
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 64
NUM_LAYERS = 2
TRAIN_RATIO = 0.8     
DROPOUT = 0.2         
TEACHER_FORCING_RATIO = 0.5  # teacher forcing during training

# Features (6 original + 4 temporal)
FEATURES = ['temperature_2m (°C)', 'relative_humidity_2m (%)', 'rain (mm)',
            'pressure_msl (hPa)', 'wind_speed_10m (km/h)', 'wind_direction_10m (°)']
TEMPORAL_FEATURES = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
ALL_FEATURES = FEATURES + TEMPORAL_FEATURES
OUTPUT_FEATURE = 'temperature_2m (°C)'

# Function to add temporal features
def add_temporal_features(df, time_col='time'):
    df[time_col] = pd.to_datetime(df[time_col])
    df['hour'] = df[time_col].dt.hour
    df['day_of_week'] = df[time_col].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    return df

# Compute normalization statistics from raw data and then normalize the data
def compute_and_apply_normalization(df, feature_cols):
    stats = {}
    df_norm = df.copy()
    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std()
        stats[col] = (mean, std)
        df_norm[col] = (df_norm[col] - mean) / std
    return df_norm, stats

# Custom Dataset for time series
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
        x = self.df.loc[idx : idx + self.input_window - 1, self.input_features].values.astype(np.float32)
        y = self.df.loc[idx + self.input_window : idx + self.input_window + self.output_window - 1, self.output_feature].values.astype(np.float32)
        y = y.reshape(-1, 1)
        return torch.tensor(x), torch.tensor(y)

# --- Models with Attention ---

# Positional Encoding for the Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Option 1: RNN with Attention (no activation at output)
class ForecastNetRNNAtt(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, output_window, dropout=0.0, use_sigmoid=False):
        super(ForecastNetRNNAtt, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder = nn.LSTM(input_size=num_features, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        self.decoder = nn.LSTM(input_size=num_features, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.Linear(hidden_size*2, 1)
        self.fc = nn.Linear(hidden_size*2, 1)
        self.proj = nn.Linear(1, num_features)
        self.output_window = output_window
        self.use_sigmoid = use_sigmoid
    def forward(self, src, target_seq=None, teacher_forcing_ratio=0.0):
        batch_size = src.size(0)
        encoder_outputs, (hidden, cell) = self.encoder(src)
        decoder_input = src[:, -1:, 0:1]  # seed: last value of 1st feature
        outputs = []
        for t in range(self.output_window):
            decoder_input_proj = self.proj(decoder_input)
            out, (hidden, cell) = self.decoder(decoder_input_proj, (hidden, cell))
            dec_hidden = out.repeat(1, encoder_outputs.size(1), 1)
            attn_input = torch.cat((encoder_outputs, dec_hidden), dim=2)
            attn_scores = self.attention(attn_input)
            attn_weights = torch.softmax(attn_scores, dim=1)
            context = torch.sum(attn_weights * encoder_outputs, dim=1, keepdim=True)
            combined = torch.cat((out, context), dim=2)
            pred = self.fc(combined)
            if self.use_sigmoid:
                pred = torch.sigmoid(pred)
            outputs.append(pred)
            if target_seq is not None and np.random.rand() < teacher_forcing_ratio:
                decoder_input = target_seq[:, t:t+1, :]
            else:
                decoder_input = pred
        outputs = torch.cat(outputs, dim=1)
        return outputs

# Option 2: Transformer for time series
class ForecastNetTransformer(nn.Module):
    def __init__(self, num_features, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, output_window, dropout=0.1, use_sigmoid=False):
        super(ForecastNetTransformer, self).__init__()
        self.d_model = d_model
        self.input_linear = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                           num_encoder_layers=num_encoder_layers,
                                           num_decoder_layers=num_decoder_layers,
                                           dim_feedforward=dim_feedforward,
                                           dropout=dropout, batch_first=True)
        self.output_linear = nn.Linear(d_model, 1)
        self.output_window = output_window
        self.use_sigmoid = use_sigmoid
    def forward(self, src, target_seq=None):
        src_emb = self.input_linear(src)
        src_emb = self.pos_encoder(src_emb)
        batch_size = src.size(0)
        # For target, use a zero sequence
        tgt = torch.zeros(batch_size, self.output_window, src.size(2), device=src.device)
        tgt_emb = self.input_linear(tgt)
        tgt_emb = self.pos_decoder(tgt_emb)
        output = self.transformer(src_emb, tgt_emb)
        output = self.output_linear(output)
        if self.use_sigmoid:
            output = torch.sigmoid(output)
        return output

# Select model type: "rnn" or "transformer"
model_type = "rnn"  # Change to "transformer" to use transformer architecture

if model_type == "rnn":
    model = ForecastNetRNNAtt(num_features=len(ALL_FEATURES), hidden_size=HIDDEN_SIZE,
                              num_layers=NUM_LAYERS, output_window=OUTPUT_WINDOW,
                              dropout=DROPOUT, use_sigmoid=False)
elif model_type == "transformer":
    model = ForecastNetTransformer(num_features=len(ALL_FEATURES), d_model=64, nhead=4,
                                   num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=128,
                                   output_window=OUTPUT_WINDOW, dropout=0.2, use_sigmoid=False)
else:
    raise ValueError("Invalid model type")

model = model.to(device)

# -----------------------------
# Load and preprocess training data
# -----------------------------
data_path = 'galapagarhoraria.csv'
df = pd.read_csv(data_path, skiprows=2, parse_dates=['time'])
df.sort_values('time', inplace=True)
df = add_temporal_features(df, time_col='time')

# Compute normalization stats on raw data and apply normalization
df_norm, stats = compute_and_apply_normalization(df.copy(), ALL_FEATURES + [OUTPUT_FEATURE])

# Create dataset and split into train/validation sets
dataset = WeatherDataset(df_norm, INPUT_WINDOW, OUTPUT_WINDOW, ALL_FEATURES, OUTPUT_FEATURE)
train_size = int(TRAIN_RATIO * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0)

# Configure training
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
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        if model_type == "rnn":
            output = model(batch_x, target_seq=batch_y, teacher_forcing_ratio=TEACHER_FORCING_RATIO)
        else:
            output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * batch_x.size(0)
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches_train:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Batch {batch_idx+1}/{total_batches_train} - Loss: {loss.item():.6f}", flush=True)
    train_loss /= len(train_dataset)
    epoch_train_losses.append(train_loss)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            if model_type == "rnn":
                output = model(batch_x)
            else:
                output = model(batch_x)
            loss = criterion(output, batch_y)
            val_loss += loss.item() * batch_x.size(0)
    val_loss /= len(val_dataset)
    epoch_val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}", flush=True)
    scheduler.step(val_loss)

# Save the model and normalization statistics
os.makedirs("model", exist_ok=True)
if model_type == "rnn":
    torch.save(model.state_dict(), "model/forecast_model_temp_rnnatt.pt")
else:
    torch.save(model.state_dict(), "model/forecast_model_temp_transformer.pt")
pd.DataFrame(stats).to_csv("model/normalization_stats_temp.csv", index=True)

plt.figure(figsize=(8,5))
plt.plot(range(1, NUM_EPOCHS+1), epoch_train_losses, marker='o', label="Training")
plt.plot(range(1, NUM_EPOCHS+1), epoch_val_losses, marker='o', label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training/Validation Loss Evolution")
plt.legend()
plt.grid(True)
plt.savefig("model/training_validation_loss_temp.png")
plt.show()

if __name__ == '__main__':
    pass
