# utils/model_utils.py
import numpy as np
from tensorflow.keras.models import load_model
import os

def extract_keypoints(results):
    """MediaPipe sonuçlarından anahtar noktaları çıkar"""
    # Poz landmark'larını ayıkla
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    
    # Yüz landmark'larını ayıkla
    face = np.array([[res.x, res.y, res.z, res.visibility] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 4)
    
    # Sol el landmark'larını ayıkla
    lh = np.array([[res.x, res.y, res.z, res.visibility] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 4)
    
    # Sağ el landmark'larını ayıkla
    rh = np.array([[res.x, res.y, res.z, res.visibility] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 4)
    
    # Tüm özellikleri birleştir
    keypoints = np.concatenate([pose, face, lh, rh])
    
    # Sabit boyuta getir (2172)
    if len(keypoints) < 2172:
        # Eksik değerleri sıfırla doldur
        keypoints = np.pad(keypoints, (0, 2172 - len(keypoints)))
    elif len(keypoints) > 2172:
        # Fazla değerleri kes
        keypoints = keypoints[:2172]
    
    return keypoints

def load_action_model(model_path):
    """Model dosyasını yükler, dosya yoksa None döndürür"""
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            print(f"Model başarıyla yüklendi: {model_path}")
            return model
        except Exception as e:
            print(f"Model yüklenirken hata oluştu: {e}")
            return None
    else:
        print(f"Model dosyası bulunamadı: {model_path}")
        return None