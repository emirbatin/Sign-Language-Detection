
import os
import cv2
import numpy as np
import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
from collections import deque
import time

# MediaPipe ayarları
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Konfigürasyon
class Config:
    MODEL_PATH = "action.h5" # Ana dizindeki model
    ACTIONS = np.array(['konnichiwa', 'arigatou', 'gomen']) # Güncel actions
    SEQUENCE_LENGTH = 30
    PREDICTION_THRESHOLD = 0.5
    PREDICTION_COOLDOWN = 1.0  

# Global değişkenler
model = None
sequence = deque(maxlen=Config.SEQUENCE_LENGTH)
predictions = deque(maxlen=10)
last_prediction_time = 0
last_prediction = ""
prediction_confidence = 0.0

def load_sign_language_model():
    """Eğitilmiş modeli yükle"""
    global model
    
    # Alternatif yollar
    paths = [
        Config.MODEL_PATH,
        os.path.join("ml-server", "models", "action.h5")
    ]
    
    for path in paths:
        if os.path.exists(path):
            try:
                model = load_model(path)
                print(f"✓ Model yüklendi: {path}")
                return True
            except Exception as e:
                print(f"❌ Model yükleme hatası ({path}): {e}")
    
    print("❌ Model dosyası bulunamadı!")
    return False

def extract_keypoints(results):
    """MediaPipe sonuçlarından anahtar noktaları çıkar - Training script ile AYNI OLMALI"""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z, 0.0] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*4)
    lh = np.array([[res.x, res.y, res.z, 0.0] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*4)
    rh = np.array([[res.x, res.y, res.z, 0.0] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*4)
    return np.concatenate([pose, face, lh, rh])

def predictions_logic(res):
    global last_prediction, prediction_confidence, last_prediction_time
    
    current_time = time.time()
    predicted_idx = np.argmax(res)
    confidence = res[predicted_idx]
    
    if confidence > Config.PREDICTION_THRESHOLD:
        if current_time - last_prediction_time > Config.PREDICTION_COOLDOWN:
             last_prediction = Config.ACTIONS[predicted_idx]
             prediction_confidence = confidence
             last_prediction_time = current_time

def predict_sign_language(frame):
    global sequence, predictions, last_prediction, prediction_confidence
    
    if frame is None:
        return None, "Kamera kapalı", "Güven: 0%", {}
    
    if model is None:
        return frame, "Model YÜKLENEMEDİ", "Lütfen modeli eğitin", {}
    
    # BGR'ye çevir
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = holistic.process(image_rgb)
        image_rgb.flags.writeable = True
        
        # Çizim (opsiyonel, Gradio'da bazen karışık görünebilir ama ekleyelim)
        mp_drawing.draw_landmarks(image_rgb, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image_rgb, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image_rgb, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        # Keypoints
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        
        probabilities = {action: 0.0 for action in Config.ACTIONS}
        
        if len(sequence) == Config.SEQUENCE_LENGTH:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            predictions_logic(res)
            
            for i, action in enumerate(Config.ACTIONS):
                probabilities[action] = float(res[i])

    processed_frame = image_rgb
    
    prediction_text = f"🧏 {last_prediction.upper()}" if last_prediction else "..."
    confidence_text = f"Güven: {prediction_confidence:.1%}"
    
    return processed_frame, prediction_text, confidence_text, probabilities

def reset_prediction():
    global sequence, last_prediction
    sequence.clear()
    last_prediction = ""
    return "", "", {}

def create_gradio_interface():
    with gr.Blocks(title="İşaret Dili Tanıma", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤟 İşaret Dili Tanıma (Holistic Model)")
        
        with gr.Row():
            with gr.Column():
                webcam = gr.Image(sources=["webcam"], streaming=True, label="Kamera")
            with gr.Column():
                pred_label = gr.Label(label="Tahmin")
                conf_label = gr.Textbox(label="Güven Skoru")
                probs_label = gr.Label(label="Olasılıklar")
                reset_btn = gr.Button("Sıfırla")
        
        webcam.stream(
            fn=predict_sign_language,
            inputs=webcam,
            outputs=[webcam, pred_label, conf_label, probs_label],
            stream_every=0.1
        )
        
        reset_btn.click(fn=reset_prediction, outputs=[pred_label, conf_label, probs_label])
        
    return demo

if __name__ == "__main__":
    if load_sign_language_model():
        demo = create_gradio_interface()
        demo.launch(server_name="0.0.0.0", server_port=7860)
