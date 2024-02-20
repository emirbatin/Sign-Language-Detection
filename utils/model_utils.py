# model_utils.py

import os
import numpy as np
from tensorflow.keras.models import load_model

def load_action_model(model_file_path):
    return load_model(model_file_path) if os.path.exists(model_file_path) else None

# Anahtar noktaları çıkarmak için fonksiyon tanımla
def extract_keypoints(results):
    # Her bir landmark için x, y, z konumlarını ve görünürlük durumunu al
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z, res.visibility] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 4)
    lh = np.array([[res.x, res.y, res.z, res.visibility] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 4)
    rh = np.array([[res.x, res.y, res.z, res.visibility] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 4)
    # Tüm landmarkları birleştir ve gerekli şekli elde etmek için tamponla veya kes
    keypoints = np.concatenate([pose, face, lh, rh])
    if len(keypoints) < 2172:
        # Uzunluk 2172'den küçükse sıfırlarla doldur
        keypoints = np.pad(keypoints, (0, 2172 - len(keypoints)))
    elif len(keypoints) > 2172:
        # Uzunluk 2172'den büyükse kes
        keypoints = keypoints[:2172]

    return keypoints

