# Gerekli kütüphaneler içe aktarılır
import sys
sys.path.append("./utils/mediapipe_utils.py")
import sys
sys.path.append("./utils/model_utils.py")
import sys
sys.path.append("./models/action_detection_model.py")

from utils.mediapipe_utils import mediapipe_detection, draw_landmarks, mp_holistic
from utils.model_utils import extract_keypoints

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
import mediapipe as mp
from  action_detection_model import actions, no_sequences

# Veri toplamak için dizinleri oluştur
def create_directories(actions):
    DATA_PATH = os.path.join("MP_Data")
    no_sequences = 5

    for action in actions:
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

# Veri toplama işlemini gerçekleştir
def collecting_data():
    # Dizinleri oluştur
    create_directories(actions)

    # Kamerayı başlat
    cap = cv2.VideoCapture(0)

    # MediaPipe Holistic modeli ile birlikte kamera akışını al
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        sequences = []  # Elde edilen özellik dizilerini saklamak için boş liste
        labels = []     # Etiketleri saklamak için boş liste
        action_index = 0    # Hareket indeksini başlat

        # Her bir hareket için veri toplama döngüsü
        while action_index < len(actions):
            action = actions[action_index]   # Mevcut hareketi al
            status_text = f"Press space to start collecting frames for {action}. Press 'q' to quit."
            print(status_text)
            messagebox.showinfo("Collecting Data", status_text)

            # Kullanıcı, veri toplamaya başlamak için boşluğa basana kadar kamera akışını göster
            while True:
                ret, frame = cap.read()
                cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', frame)
                if cv2.waitKey(1) & 0xFF == ord(' '):
                    break
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            sequence_index = 0  # Özellik dizisi indeksini sıfırla
            # Belirli bir hareketin bir dizi görüntüsünü toplama döngüsü
            while sequence_index < no_sequences:
                window = []  # Görüntü penceresini saklamak için boş liste
                frame_num = 0   # Görüntü numarasını sıfırla

                # 30 karelik bir pencere toplama döngüsü
                while frame_num < 30:
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    image, results = mediapipe_detection(frame, holistic)
                    draw_landmarks(image, results)

                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (50, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, f'Collecting frames for {action} Video number {sequence_index + 1}', (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(image, f'Collecting frames for {action} Video number {sequence_index + 1}', (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                    keypoints = extract_keypoints(results)

                    if np.shape(keypoints) != (2172,):
                        print(f"Incompatible frame size: {np.shape(keypoints)}")
                        continue

                    npy_path = os.path.join("MP_Data", action, str(sequence_index), str(frame_num))
                    np.save(npy_path, keypoints)

                    window.append(keypoints)
                    cv2.imshow('OpenCV Feed', image)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

                    frame_num += 1

                sequences.append(window)  # Elde edilen özellik dizilerini listeye ekle
                labels.append(action)     # Etiketleri listeye ekle

                sequence_index += 1

            print(f"Recording for {action} completed. Press space to start collecting frames for the next word.")

            # Kullanıcı bir sonraki kelimenin toplanmasına başlamak için boşluğa basana kadar kamera akışını göster
            while True:
                ret, frame = cap.read()
                cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', frame)
                if cv2.waitKey(1) & 0xFF == ord(' '):
                    break
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            action_index += 1

    cap.release()   # Kamerayı serbest bırak
    cv2.destroyAllWindows()  # Pencereleri kapat
