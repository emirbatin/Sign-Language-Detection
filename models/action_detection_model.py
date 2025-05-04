# models/action_detection_model.py
import os
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, BatchNormalization
from tensorflow.keras import regularizers

# Temel değişkenler
DATA_PATH = os.path.join("MP_Data")
actions = np.array(['konnichiwa', 'arigatou', 'gomen', 'suki', 'nani', 'daijoubu', 'namae', 'genki'])
no_sequences = 30
sequence_length = 30

# Model dosya yolu
model_file_path = 'ML_Models/action.keras'

# Model yükle (eğer dosya varsa)
model = None
if os.path.exists(model_file_path):
    try:
        model = load_model(model_file_path)
        print("Model başarıyla yüklendi.")
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {e}")

def create_model():
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(30, 2172)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    return model