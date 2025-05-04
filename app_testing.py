# app_testing.py
import os
import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.models import load_model

from models.action_detection_model import actions
from utils.mediapipe_utils import mediapipe_detection, draw_landmarks, mp_holistic
from utils.model_utils import extract_keypoints

def test_app(model, actions):
    """Eğitilmiş modeli test eder tasarımlı arayüzle gerçek zamanlı tahminler yapar"""
    # Tk root penceresi oluştur (messagebox için gerekli)
    root = tk.Tk()
    root.withdraw()  # Pencereyi gizle
    
    # Model kontrolü
    model_file_path = 'ML_Models/action.keras'
    if model is None:
        if not os.path.exists(model_file_path):
            messagebox.showerror("Hata", "Model bulunamadı. Lütfen önce modeli eğitin.")
            return
        
        # Model dosyası varsa yükle
        model = load_model(model_file_path)
        print("Model başarıyla yüklendi.")
    
    # Gerekli değişkenler
    sequence = []
    predictions = []
    sentence = []
    threshold = 0.4
    
    # Son tahmin zamanı ve animasyon için değişkenler
    last_prediction_time = time.time()
    highlight_effect = 0
    fade_speed = 5  # Vurgu efekti hızı
    prediction_cooldown = 1.0  # Tahminler arasındaki minimum süre (saniye)
    
    # renk paleti
    colors = {
        "primary": (76, 80, 213),     # Zengin mavi
        "secondary": (243, 176, 49),  # Altın sarısı
        "bg_dark": (32, 33, 36),      # Koyu gri
        "bg_light": (60, 64, 67),     # Orta gri
        "text_light": (255, 255, 255),# Beyaz
        "accent": (32, 203, 111),     # Turkuaz
        "warning": (66, 66, 212)      # Turuncu
    }
    
    # Uygulama başlık bilgisi
    app_title = "Isaret Dili Tanima"
    
    # Kamerayı başlat
    cap = cv2.VideoCapture(1)
    
    # Ekran boyutlarını ayarla
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Ekran oranını hesapla
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Blur efekti için kernel
    blur_kernel = np.ones((5,5), np.float32) / 25
    
    # MediaPipe Holistic modeli ile birlikte kamera akışını al
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Kamera görüntüsü alınamadı.")
                continue
            
            # Görüntüyü ayna modunda çevir
            frame = cv2.flip(frame, 1)
            
            # Orijinal görüntüyü kopyala
            original_image = frame.copy()
            
            # MediaPipe ile el, yüz ve poz tespiti
            image, results = mediapipe_detection(frame, holistic)
            
            # Algılanan landmark'ları çiz
            draw_landmarks(image, results)
            
            # arka plan oluştur
            # Modern tarzda yarı saydam katman ekle
            overlay = image.copy()
            
            # Üst bilgi çubuğu
            cv2.rectangle(overlay, (0, 0), (frame_width, 80), colors["bg_dark"], -1)
            cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
            
            # Uygulama başlığı
            cv2.putText(image, app_title, (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors["secondary"], 2, cv2.LINE_AA)
            
            # Tahmin paneli
            cv2.rectangle(overlay, (frame_width-400, 100), (frame_width-50, 400), colors["bg_dark"], -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
            
            # Tahmin paneli başlığı
            cv2.putText(image, "TANIMA SONUCLARI", (frame_width-375, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["secondary"], 2, cv2.LINE_AA)
            
            # Alt bilgi çubuğu
            cv2.rectangle(overlay, (0, frame_height-60), (frame_width, frame_height), colors["bg_dark"], -1)
            cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
            
            # Alt bilgi metni
            status_text = "El hareketi tanima aktif | Cikmak icin 'q' tusuna basin"
            cv2.putText(image, status_text, (20, frame_height-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["text_light"], 1, cv2.LINE_AA)
            
            # Hareket tahmini
            current_time = time.time()
            if results.left_hand_landmarks or results.right_hand_landmarks:
                # Anahtar noktaları çıkar
                keypoints = extract_keypoints(results)
                
                # Diziye ekle ve son 30 kareyi tut
                sequence.append(keypoints)
                sequence = sequence[-30:]
                
                # Dizi tam 30 kare olduğunda tahmin yap
                if len(sequence) == 30:
                    # Model tahmini
                    input_data = np.expand_dims(sequence, axis=0)
                    result = model.predict(input_data)[0]
                    
                    # En yüksek olasılıklı sınıfı bul
                    predicted_class_idx = np.argmax(result)
                    predicted_class = actions[predicted_class_idx]
                    confidence = result[predicted_class_idx]
                    
                    # Tahminleri kaydet
                    predictions.append(predicted_class_idx)
                    
                    # Yeni bir tahmin yapıldıysa
                    if current_time - last_prediction_time > prediction_cooldown:
                        # Eşik değeri kontrolü
                        if confidence > threshold:
                            # Son 10 tahmini kontrol et
                            if len(predictions) > 10:
                                # En sık tekrarlanan sınıfı bul
                                unique_classes, counts = np.unique(predictions[-10:], return_counts=True)
                                mode_class_idx = unique_classes[np.argmax(counts)]
                                
                                # Tutarlı tahmini teyit et
                                if mode_class_idx == predicted_class_idx:
                                    # Eğer yeni bir kelime ise ekleyelim
                                    if len(sentence) == 0 or predicted_class != sentence[-1]:
                                        sentence.append(predicted_class)
                                        print(f"Tahmin: {predicted_class}, Guven: {confidence:.4f}")
                                        
                                        # Yeni tahmin zamanını güncelle
                                        last_prediction_time = current_time
                                        highlight_effect = 1.0  # Vurgu efektini başlat
            
            # Son 3 tahmini göster
            sentence = sentence[-3:]
            
            # Efekt faktörünü güncelle (zamanla azalt)
            highlight_effect = max(0, highlight_effect - (fade_speed * (current_time - last_prediction_time)))
            
            # Tahmin sonuçlarını göster
            if len(sentence) > 0:
                # Sonuç etiketi
                cv2.putText(image, "Algilanan Isaret:", (frame_width-375, 170), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["text_light"], 1, cv2.LINE_AA)
                
                # Son tahmini göster (vurgu efektiyle)
                result_color = colors["secondary"]
                # Son tahminin renk geçişi (highlight_effect değerine göre)
                if highlight_effect > 0:
                    intensity = int(255 * highlight_effect)
                    result_color = (intensity, intensity, 255)  # Parlayan beyaza doğru git
                
                # Mevcut tahmin içn büyük metin
                cv2.putText(image, sentence[-1].upper(), (frame_width-375, 240), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.5, result_color, 2, cv2.LINE_AA)
                
                # Önceki tahminler için (daha küçük)
                if len(sentence) > 1:
                    prev_text = " - ".join(sentence[:-1])
                    cv2.putText(image, f"Onceki: {prev_text}", (frame_width-375, 300), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["text_light"], 1, cv2.LINE_AA)
            else:
                cv2.putText(image, "Hareket bekleniyor...", (frame_width-375, 240), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors["text_light"], 1, cv2.LINE_AA)
            
            # Güven çubuğu göster (eğer tahmin varsa)
            if len(sequence) == 30 and len(sentence) > 0:
                confidence = result[predicted_class_idx]
                
                # Çubuk başlangıç noktası
                bar_x = frame_width-375
                bar_y = 340
                bar_width = 300
                bar_height = 30
                
                # Arka plan çubuğu
                cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                             colors["bg_light"], -1)
                
                # Güven seviyesi çubuğu
                filled_width = int(bar_width * confidence)
                
                # Güven seviyesine göre renk belirle
                bar_color = colors["warning"]  # Varsayılan turuncu
                if confidence > 0.7:
                    bar_color = colors["accent"]  # Yüksek güven - yeşil
                
                cv2.rectangle(image, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), 
                             bar_color, -1)
                
                # Çubuk kenarı
                cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                             colors["text_light"], 1)
                
                # Güven yüzdesi
                cv2.putText(image, f"Guven: %{int(confidence*100)}", (bar_x + 10, bar_y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["text_light"], 1, cv2.LINE_AA)
            
            # Ana görüntü çerçevesi
            cv2.rectangle(image, (10, 90), (frame_width-420, frame_height-70), colors["primary"], 2)
            
            # Zaman damgası
            time_str = time.strftime("%H:%M:%S")
            cv2.putText(image, time_str, (frame_width-150, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["text_light"], 1, cv2.LINE_AA)
            
            # Logo/marka alanı
            logo_text = "ISARET AI"
            cv2.putText(image, logo_text, (frame_width-150, frame_height-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["secondary"], 2, cv2.LINE_AA)
            
            # Kamera görüntüsünü göster
            cv2.imshow('Isaret Dili Tanima', image)
            
            # Çıkış kontrolü
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        # Kamerayı serbest bırak ve pencereleri kapat
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Modeli yükle
    model_file_path = 'ML_Models/action.keras'
    model = load_model(model_file_path) if os.path.exists(model_file_path) else None
    
    # Test uygulamasını başlat
    test_app(model, actions)