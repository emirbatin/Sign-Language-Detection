# data_collecting.py
import os
import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import messagebox

# MacOS M2 optimizasyonları
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # TensorFlow uyarı mesajlarını azalt

# Modülleri içe aktar
import sys
sys.path.append("./models")
sys.path.append("./utils")

from models.action_detection_model import actions, no_sequences, sequence_length, DATA_PATH
from utils.mediapipe_utils import mediapipe_detection, draw_landmarks, mp_holistic
from utils.model_utils import extract_keypoints

def create_directories():
    """Veri toplama için gerekli dizinleri oluşturur"""
    os.makedirs(DATA_PATH, exist_ok=True)
    
    for action in actions:
        action_dir = os.path.join(DATA_PATH, action)
        os.makedirs(action_dir, exist_ok=True)
        
        for sequence in range(no_sequences):
            seq_dir = os.path.join(action_dir, str(sequence))
            os.makedirs(seq_dir, exist_ok=True)

def collecting_data():
    """İşaret dili veri toplama - performans optimize edilmiş"""
    print("=== İşaret Dili Veri Toplayıcı ===")
    
    # Dizinleri oluştur
    create_directories()
    
    # Kamera bağlantısını bir kez oluştur (daha yüksek performans için)
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Kamera açılamadı!")
        return
    
    # MediaPipe modeli
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Her hareket için
        for action_idx, action in enumerate(actions):
            print(f"\n[HAREKET {action_idx+1}/{len(actions)}] '{action}' için veri toplama")
            
            # Her sekans için
            for sequence in range(no_sequences):
                print(f"\n--- Video {sequence+1}/{no_sequences} ---")
                print(f"'{action}' hareketi için hazırlanın. Başlamak için kamera penceresindeyken ENTER tuşuna basın...")
                
                # Kullanıcı ENTER'a basana kadar normal kamera görüntüsünü göster
                waiting_for_enter = True
                while waiting_for_enter:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    # Bilgi yazısı
                    cv2.putText(frame, f"'{action}' - Video {sequence+1}/{no_sequences}", (15, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, "Başlamak için ENTER, çıkmak için 'q' tuşuna basın", (15, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Görüntüyü göster
                    cv2.imshow('Kamera', frame)
                    
                    # Tuş kontrolü
                    key = cv2.waitKey(1)
                    if key == 13:  # ENTER tuşu
                        waiting_for_enter = False
                    elif key == ord('q'):
                        print("Veri toplama kullanıcı tarafından iptal edildi.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                
                # Geri sayım
                for countdown in range(3, 0, -1):
                    print(f"{countdown}...")
                    
                    # Her sayı için birkaç kare göster
                    start_time = time.time()
                    while time.time() - start_time < 1.0:
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        
                        # Sayı yazısı
                        cv2.putText(frame, str(countdown), (320, 240), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5, cv2.LINE_AA)
                        
                        # Görüntüyü göster
                        cv2.imshow('Kamera', frame)
                        cv2.waitKey(1)
                
                # Kayıt başlıyor
                print("KAYIT BAŞLADI!")
                
                # Kare toplamaya başla
                frame_count = 0
                while frame_count < sequence_length:
                    # Kare oku
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    # MediaPipe işleme
                    try:
                        # Görüntüyü işle
                        image, results = mediapipe_detection(frame, holistic)
                        draw_landmarks(image, results)
                        
                        # Ekrana bilgi yazdır
                        cv2.putText(image, "KAYIT YAPILIYOR", (20, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                        
                        cv2.putText(image, f"{action} - Video {sequence+1}/{no_sequences}", (20, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                        
                        cv2.putText(image, f"Kare: {frame_count+1}/{sequence_length}", (20, 90), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                        
                        # Görüntüyü göster
                        cv2.imshow('Kamera', image)
                        
                        # Anahtar noktaları çıkar
                        keypoints = extract_keypoints(results)
                        
                        # Boyut kontrolü
                        if np.shape(keypoints) != (2172,):
                            print(f"Uyumsuz kare boyutu: {np.shape(keypoints)}")
                            continue
                        
                        # Kaydet
                        npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_count))
                        np.save(npy_path, keypoints)
                        
                        # Sonraki kare
                        frame_count += 1
                        
                        # İlerleme
                        if frame_count % 10 == 0:
                            print(f"İlerleme: {frame_count}/{sequence_length} kare")
                        
                        # Çıkış kontrolü
                        if cv2.waitKey(1) == ord('q'):
                            print("Veri toplama kullanıcı tarafından iptal edildi.")
                            cap.release()
                            cv2.destroyAllWindows()
                            return
                    
                    except Exception as e:
                        print(f"Kare işleme hatası: {e}")
                        continue
                
                print(f"Video {sequence+1}/{no_sequences} tamamlandı!")
                
                # Bir sonraki video için devam etmek istiyor mu?
                if sequence < no_sequences - 1:
                    # Devam mesajını göster
                    for i in range(3):  # 3 saniye göster
                        start_time = time.time()
                        while time.time() - start_time < 1.0:
                            ret, frame = cap.read()
                            if not ret:
                                continue
                            
                            cv2.putText(frame, "Video tamamlandı!", (20, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                            cv2.putText(frame, "Devam etmek için ENTER, çıkmak için 'q'", (20, 60), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                            
                            cv2.imshow('Kamera', frame)
                            
                            key = cv2.waitKey(1)
                            if key == 13:
                                break
                            elif key == ord('q'):
                                print("Veri toplama kullanıcı tarafından iptal edildi.")
                                cap.release()
                                cv2.destroyAllWindows()
                                return
                    
                    # Kısa bir süre bekle
                    time.sleep(0.5)
            
            print(f"'{action}' hareketi için tüm videolar tamamlandı!")
            
            # Bir sonraki harekete geçmek istiyor mu?
            if action_idx < len(actions) - 1:
                next_action = actions[action_idx + 1]
                
                # Devam mesajını göster
                for i in range(5):  # 5 saniye göster
                    start_time = time.time()
                    while time.time() - start_time < 1.0:
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        
                        cv2.putText(frame, f"'{action}' tamamlandı!", (20, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, f"Sıradaki: '{next_action}'", (20, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(frame, "Devam etmek için ENTER, çıkmak için 'q'", (20, 90), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                        
                        cv2.imshow('Kamera', frame)
                        
                        key = cv2.waitKey(1)
                        if key == 13:
                            break
                        elif key == ord('q'):
                            print("Veri toplama kullanıcı tarafından iptal edildi.")
                            cap.release()
                            cv2.destroyAllWindows()
                            return
                
                # Kısa bir süre bekle
                time.sleep(0.5)
        
        print("\nTüm veri toplama işlemi tamamlandı!")
    
    # Kaynakları serbest bırak
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collecting_data()