#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de entrenamiento para la predicción meteorológica
utilizando una arquitectura encoder-decoder con LSTM, atención y técnicas de regularización.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Concatenate, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import regularizers
import tensorflow as tf

# Parámetros del modelo y de la serie temporal
INPUT_LENGTH = 168   # 7 días de historia (horas)
OUTPUT_LENGTH = 72   # Predicción para 72 horas
NUM_FEATURES = 5     # Variables: temperatura, humedad, lluvia, presión y velocidad del viento
LATENT_DIM = 64      # Dimensión del espacio latente
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4  # Se reduce ligeramente el learning rate para estabilizar el entrenamiento

# 1. Preprocesamiento: Lectura y limpieza de datos
def load_data(csv_file):
    """
    Carga el CSV omitiendo las dos primeras líneas de metadatos,
    parsea la columna de tiempo y selecciona las 5 variables clave.
    """
    # Leer CSV saltando las dos primeras líneas
    df = pd.read_csv(csv_file, skiprows=2)
    # Asegurar que la columna 'time' se convierta a datetime
    df['time'] = pd.to_datetime(df['time'])
    # Seleccionar las variables de interés
    columnas = ["temperature_2m (°C)", "relative_humidity_2m (%)", 
                "rain (mm)", "pressure_msl (hPa)", "wind_speed_10m (km/h)"]
    df = df[['time'] + columnas]
    # Convertir las columnas a valores numéricos (si fuera necesario)
    for col in columnas:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    return df

# 2. Creación de secuencias para Entrenamiento
def create_sequences(data, input_len=INPUT_LENGTH, output_len=OUTPUT_LENGTH):
    """
    Genera pares de secuencias (X, y) a partir de los datos escalados.
    X tendrá forma (n_muestras, input_len, NUM_FEATURES)
    y tendrá forma (n_muestras, output_len, NUM_FEATURES)
    """
    X, y = [], []
    for i in range(len(data) - input_len - output_len + 1):
        X.append(data[i : i + input_len])
        y.append(data[i + input_len : i + input_len + output_len])
    return np.array(X), np.array(y)

# 3. Definición de la capa de Atención personalizada
class AttentionLayer(Layer):
    """
    Capa de atención que combina la secuencia del decodificador
    con la secuencia del codificador para generar un vector de contexto.
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # input_shape: [decoder_outputs, encoder_outputs]
        # decoder_outputs: (batch, T_dec, latent_dim)
        # encoder_outputs: (batch, T_enc, latent_dim)
        self.W = self.add_weight(name="att_weight",
                                 shape=(input_shape[0][-1], input_shape[0][-1]),
                                 initializer="glorot_uniform",
                                 trainable=True)
        self.V = self.add_weight(name="att_v",
                                 shape=(input_shape[0][-1], 1),
                                 initializer="glorot_uniform",
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        decoder_outputs, encoder_outputs = inputs
        # Transformar la salida del decodificador
        decoder_transformed = tf.tensordot(decoder_outputs, self.W, axes=[[2], [0]])
        # Expandir dimensiones para la suma con las salidas del codificador
        decoder_expanded = tf.expand_dims(decoder_transformed, 2)
        encoder_expanded = tf.expand_dims(encoder_outputs, 1)
        # Sumar y aplicar tangente hiperbólica
        score = tf.nn.tanh(decoder_expanded + encoder_expanded)
        # Proyectar a una dimensión
        score = tf.tensordot(score, self.V, axes=[[3], [0]])
        # Eliminar dimensión extra: (batch, T_dec, T_enc)
        score = tf.squeeze(score, -1)
        # Calcular pesos de atención con softmax (sobre T_enc)
        attention_weights = tf.nn.softmax(score, axis=-1)
        # Calcular vector de contexto como suma ponderada de las salidas del codificador
        context_vector = tf.matmul(attention_weights, encoder_outputs)
        return context_vector

# 4. Construcción del modelo encoder-decoder con atención y regularización
def build_model(input_len=INPUT_LENGTH, output_len=OUTPUT_LENGTH, num_features=NUM_FEATURES, latent_dim=LATENT_DIM):
    # Entrada del codificador
    encoder_inputs = Input(shape=(input_len, num_features))
    # Capa LSTM del codificador con dropout y recurrent_dropout
    encoder_lstm, state_h, state_c = LSTM(latent_dim, 
                                          dropout=0.2,
                                          recurrent_dropout=0.1,
                                          return_sequences=True, 
                                          return_state=True,
                                          kernel_regularizer=regularizers.l2(1e-4)
                                         )(encoder_inputs)
    encoder_outputs = encoder_lstm  # Salida completa del codificador
    
    # Decodificador: Se utiliza el último estado del codificador y se repite para cada paso de salida
    decoder_inputs = RepeatVector(output_len)(state_h)
    # Capa LSTM del decodificador con dropout y recurrent_dropout
    decoder_lstm = LSTM(latent_dim, 
                        dropout=0.2,
                        recurrent_dropout=0.1,
                        return_sequences=True,
                        kernel_regularizer=regularizers.l2(1e-4))
    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])
    
    # Mecanismo de Atención: Se utiliza la salida del decodificador y la secuencia del codificador
    context_vector = AttentionLayer()([decoder_outputs, encoder_outputs])
    # Combinar la salida del decodificador con el vector de contexto
    decoder_combined_context = Concatenate(axis=-1)([decoder_outputs, context_vector])
    
    # Capa final que predice para cada uno de los 'output_len' pasos y 'num_features' variables
    decoder_dense = TimeDistributed(Dense(num_features))
    outputs = decoder_dense(decoder_combined_context)
    
    model = Model(encoder_inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mse')
    model.summary()
    return model

# 5. Proceso principal de entrenamiento
def main():
    # Cargar datos históricos (por ejemplo, "galapagarhoraria.csv")
    df = load_data('galapagarhoraria.csv')
    print("Datos cargados:", df.shape)
    
    # Seleccionar únicamente las variables (excluyendo la columna de tiempo)
    data_values = df.drop('time', axis=1).values

    # Escalado de los datos
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
    
    # Construir el modelo
    model = build_model()
    
    # Callbacks: aumentar la paciencia a 10 épocas y guardar el mejor modelo
    checkpoint = ModelCheckpoint('weather_forecast_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    
    # Entrenamiento
    history = model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val),
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        callbacks=[checkpoint, early_stop])
    
    # Graficar la evolución de la pérdida
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Loss entrenamiento')
    plt.plot(history.history['val_loss'], label='Loss validación')
    plt.xlabel('Época')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('Pérdida durante el entrenamiento')
    plt.savefig('training_loss.png')
    plt.show()

if __name__ == '__main__':
    main()
