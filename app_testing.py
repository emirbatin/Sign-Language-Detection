# app_testing.py dosyası

# Gerekli kütüphaneler ve modüller içe aktarılır
import sys
sys.path.append("./utils/mediapipe_utils.py")
import sys
sys.path.append("./utils/model_utils.py")

from utils.mediapipe_utils import mediapipe_detection, draw_landmarks, mp_holistic
from utils.model_utils import extract_keypoints

import os
from tkinter import messagebox
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model


# Diğer fonksiyonları içe aktarabilirsiniz

# Uygulamanın test edilmesi için bir fonksiyon tanımlanır
def test_app(model, actions):
    # Eğer model yoksa
    if model is None:
        # Model dosyasının varlığını kontrol et
        if not os.path.exists(model_file_path):
            messagebox.showerror("Error", "Model file not found. Please train the model first.")
            return

        # Model dosyası varsa yükle
        model = load_model(model_file_path)
        print("Model loaded successfully.")
    
    # Gerekli değişkenler tanımlanır
    sequence = []
    res = []
    sentence = []
    predictions = []
    threshold = 0.4

    # MediaPipe kütüphanesi kullanılarak el hareketlerinin algılanması için Holistic modeli başlatılır
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Kamera başlatılır
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # El hareketlerinin algılanması
            image, results = mediapipe_detection(frame, holistic)

            # Algılanan el hareketlerinin çizdirilmesi
            draw_landmarks(image, results)

            # Eğer el görünüyorsa
            if results.left_hand_landmarks or results.right_hand_landmarks:
                # El noktaları çıkarılır
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]  # Son 30 kareyi saklar

                # El hareketi dizisi oluşturulur ve model tarafından tahmin yapılır
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
                    predictions.append(np.argmax(res))
            else:
                # Eğer el görünmüyorsa
                predictions.append(len(actions) - 1)

            # Son 10 tahmin arasında benzersiz bir tahmin varsa ve en yüksek tahmin belirlenen eşik değerini aşıyorsa
            if len(predictions) > 0 and len(res) > 0 and np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)].any() > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[:1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 1:
                sentence = sentence[-1:]

            # Ekran üzerine son tahminin yazdırılması
            cv2.rectangle(image, (0, 0), (500, 80), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 2,
                        cv2.LINE_AA)

            # Ekran gösterimi
            cv2.imshow('OpenCV Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        # Kamera serbest bırakılır ve pencereler kapatılır
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Model dosya yolu ve hareket etiketleri tanımlanır
    model_file_path = 'ML_Models/action.keras'
    actions = np.array(['hello', 'thanks', 'howareyou'])

    # Test uygulaması başlatılır ve model yüklenir
    test_app(load_model(model_file_path))
