import os
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import tensorflow as tf

#Verilerin çıkarılacağı yol, numpys arrays
DATA_PATH = os.path.join("MP_Data")

#Harketlerimizi tespit etmeye çalışacak
actions = np.array(['hello','thanks','howareyou'])

#30 Video değerinde verilerimiz olacak
no_sequences = 30

#Video 30 kare uzunluğunda olacak 
sequence_length = 30

# Eğer action.keras dosyası mevcut değilse model değişkenini None olarak ayarla
model_file_path = 'ML_Models/action.keras'
model = load_model(model_file_path) if os.path.exists(model_file_path) else None

# Şimdi model değişkeni None değerine sahip olacak, çünkü dosya bulunmuyor
print(model)


""" # Fonksiyonları tanımla
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR'dan RGB'ye dönüştür
    image.flags.writeable = False                   # Görüntüyü salt okunur yap
    results = model.process(image)                  # Modelle algılama yap
    image.flags.writeable = True                    # Görüntüyü tekrar yazılabilir yap
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # RGB'den BGR'ye dönüştür
    return image, results

#Default Landmarks
def draw_landmarks(image, results):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp.solutions.holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
        
#Custom Landmarks (!)        
def draw_styled_landmarks(image, results):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Yüz bağlantılarını çiz
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            results.face_landmarks, 
            mp_holistic.FACEMESH_TESSELATION,  # Yüz landmarklarının bağlantıları için
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
        )
        
    # Vücut pozisyon bağlantılarını çiz
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            mp_holistic.POSE_CONNECTIONS,  # Vücut pozisyonlarının bağlantıları için
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
        )
        
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            results.left_hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS, # Sol el pozisyonlarının bağlantıları için
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=2, circle_radius=1),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
        )
        
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS, # Sağ el pozisyonlarının bağlantıları için
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=2, circle_radius=1),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
        )


#mp_holistic.POSE_CONNECTIONS

#mp_drawing.draw_landmarks??


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z, res.visibility] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 4)
    lh = np.array([[res.x, res.y, res.z, res.visibility] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 4)
    rh = np.array([[res.x, res.y, res.z, res.visibility] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 4)
    # Anahtar noktaları birleştirin ve istenen şekli elde etmek için tamponlayın veya kesin
    keypoints = np.concatenate([pose, face, lh, rh])
    if len(keypoints) < 2172:
        # Uzunluk 2172'den azsa sıfırlarla doldur
        keypoints = np.pad(keypoints, (0, 2172 - len(keypoints)))
    elif len(keypoints) > 2172:
        # Uzunluk 2172'den büyükse kes
        keypoints = keypoints[:2172]

    return keypoints
 """