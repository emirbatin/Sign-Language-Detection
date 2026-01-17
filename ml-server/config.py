
# Model Configuration
import numpy as np

# Aynı sırayla olmalı!
ACTIONS = np.array(['konnichiwa', 'arigatou', 'gomen'])

# Model ayarları
MODEL_PATH = 'models/action.h5'
SEQUENCE_LENGTH = 30
PREDICTION_THRESHOLD = 0.7
