# utils/mediapipe_utils.py
import cv2
import mediapipe as mp
import os

# MacOS M2 optimizasyonları
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # TensorFlow uyarı mesajlarını azalt

# MediaPipe modelleri
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def create_capture(camera_index=0, width=640, height=480, fps=30):
    """
    MacOS M2 uyumlu kamera bağlantısı oluşturur.
    Her çağrıda yeni bir capture nesnesi döner.
    """
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap

def mediapipe_detection(image, model):
    """MediaPipe modeli kullanarak görüntüde algılama yapar"""
    # BGR'den RGB'ye dönüştür (MediaPipe RGB bekler)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False  # Performans için görüntü salt okunur yap
    results = model.process(image)  # İşleme
    image.flags.writeable = True  # Görüntüyü tekrar yazılabilir yap
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # RGB'den BGR'ye geri dönüştür
    return image, results

def draw_landmarks(image, results):
    """Algılanan landmark'ları varsayılan stillerle çizer"""
    # Yüz landmark'larını çiz
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
    
    # Poz landmark'larını çiz
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    # Sol el landmark'larını çiz
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
    
    # Sağ el landmark'larını çiz
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())