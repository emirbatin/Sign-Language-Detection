# utils/visualization_utils.py
import cv2
from utils.mediapipe_utils import mp_holistic, mp_drawing

def draw_styled_landmarks(image, results):
    """Algılanan landmark'ları özelleştirilmiş stillerle çizer"""
    # Yüz landmark'larını çiz
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
        )
    
    # Poz landmark'larını çiz
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
        )
    
    # Sol el landmark'larını çiz
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=2, circle_radius=1),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
        )
    
    # Sağ el landmark'larını çiz
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=2, circle_radius=1),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
        )

def create_premium_ui(image, sentence, result=None, predicted_class_idx=None, confidence=None, highlight_effect=0, colors=None):
    """ UI ekler"""
    if colors is None:
        colors = {
            "primary": (76, 80, 213),     # Zengin mavi
            "secondary": (243, 176, 49),  # Altın sarısı
            "bg_dark": (32, 33, 36),      # Koyu gri
            "bg_light": (60, 64, 67),     # Orta gri
            "text_light": (255, 255, 255),# Beyaz
            "accent": (32, 203, 111),     # Turkuaz
            "warning": (66, 66, 212)      # Turuncu
        }
    
    height, width = image.shape[:2]
    
    # Overlay için kopyala
    overlay = image.copy()
    
    # Üst bilgi çubuğu
    cv2.rectangle(overlay, (0, 0), (width, 80), colors["bg_dark"], -1)
    cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
    
    # Uygulama başlığı
    cv2.putText(image, "Isaret Dili Tanima", (20, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors["secondary"], 2, cv2.LINE_AA)
    
    # Tahmin paneli
    cv2.rectangle(overlay, (width-400, 100), (width-50, 400), colors["bg_dark"], -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    # Tahmin paneli başlığı
    cv2.putText(image, "TANIMA SONUCLARI", (width-375, 130), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["secondary"], 2, cv2.LINE_AA)
    
    # Alt bilgi çubuğu
    cv2.rectangle(overlay, (0, height-60), (width, height), colors["bg_dark"], -1)
    cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
    
    # Alt bilgi metni
    status_text = "El hareketi tanima aktif | Çikmak için 'q' tuşuna basın"
    cv2.putText(image, status_text, (20, height-25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["text_light"], 1, cv2.LINE_AA)
    
    # Tahmin sonuçlarını göster
    if len(sentence) > 0:
        # Sonuç etiketi
        cv2.putText(image, "Algilanan Isaret:", (width-375, 170), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["text_light"], 1, cv2.LINE_AA)
        
        # Son tahmini göster (vurgu efektiyle)
        result_color = colors["secondary"]
        # Son tahminin renk geçişi (highlight_effect değerine göre)
        if highlight_effect > 0:
            intensity = int(255 * highlight_effect)
            result_color = (intensity, intensity, 255)  # Parlayan beyaza doğru git
        
        # Mevcut tahmin için büyük metin
        cv2.putText(image, sentence[-1].upper(), (width-375, 240), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1.5, result_color, 2, cv2.LINE_AA)
        
        # Önceki tahminler için (daha küçük)
        if len(sentence) > 1:
            prev_text = " - ".join(sentence[:-1])
            cv2.putText(image, f"Onceki: {prev_text}", (width-375, 300), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["text_light"], 1, cv2.LINE_AA)
    else:
        cv2.putText(image, "Hareket bekleniyor...", (width-375, 240), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors["text_light"], 1, cv2.LINE_AA)
    
    # Güven çubuğu göster (eğer tahmin varsa)
    if result is not None and confidence is not None:
        # Çubuk başlangıç noktası
        bar_x = width-375
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
    cv2.rectangle(image, (10, 90), (width-420, height-70), colors["primary"], 2)
    
    # Zaman damgası
    import time
    time_str = time.strftime("%H:%M:%S")
    cv2.putText(image, time_str, (width-150, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["text_light"], 1, cv2.LINE_AA)
    
    # Logo/marka alanı
    logo_text = "ISARET AI"
    cv2.putText(image, logo_text, (width-150, height-25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["secondary"], 2, cv2.LINE_AA)
    
    return image