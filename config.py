# config.py
import os
import numpy as np

class Config:
    # Yollar
    DATA_PATH = "MP_Data"
    MODEL_DIR = "ML_Models"
    KERAS_MODEL_PATH = os.path.join(MODEL_DIR, "action.keras")
    TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "action.tflite")
    
    # Model parametreleri
    ACTIONS = np.array(['konnichiwa', 'arigatou', 'gomen', 'suki', 'nani', 'daijoubu', 'namae', 'genki'])
    NO_SEQUENCES = 30
    SEQUENCE_LENGTH = 30
    
    # Eğitim parametreleri
    TRAIN_TEST_SPLIT = 0.2
    VALIDATION_SPLIT = 0.1
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    MAX_EPOCHS = 1000
    EARLY_STOPPING_PATIENCE = 5
    
    # Test parametreleri
    PREDICTION_THRESHOLD = 0.4
    
    # MacOS M2 Optimizasyonları
    USE_PER_FRAME_CAPTURE = True  # Her kare için kamerayı yeniden başlat
    CAMERA_INDEX = 0  # Kamera indeksi
    FRAME_WIDTH = 640  # Kare genişliği
    FRAME_HEIGHT = 480  # Kare yüksekliği
    FRAME_RATE = 30  # FPS