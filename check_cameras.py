
import cv2

def check_cameras():
    print("🎥 Kapsamlı Kamera Kontrolü Başlatılıyor...\n")
    
    # Test edilecek backend'ler
    backends = [
        (cv2.CAP_ANY, "Default (Auto)"),
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation")
    ]
    
    found_any = False
    
    for backend_id, backend_name in backends:
        print(f"--- Testing Backend: {backend_name} ---")
        for index in range(3): # İlk 3 portu dene
            cap = cv2.VideoCapture(index, backend_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"✅ Kamera ÇALIŞTI! -> Index: {index}, Backend: {backend_name}")
                    found_any = True
                    cap.release()
                else:
                    print(f"❌ Index {index}: Açıldı ama görüntü alınamadı (Frame Yok)")
                    cap.release()
            else:
                print(f"❌ Index {index}: Bağlantı başarısız")
        print("")
            
    print("="*40)
    if found_any:
        print("✅ Çalışan bir kombinasyon bulundu! Yukarıdaki 'Index' ve 'Backend' bilgisini kullanalım.")
    else:
        print("⚠️ Hâlâ hiç kamera bulunamadı.")
        print("Çözüm Önerileri:")
        print("1. Camo Studio'yu tamamen kapatıp (System Tray dahil) tekrar yönetici olarak açın.")
        print("2. Başka bir programın (Zoom, Discord vb.) kamerayı kullanmadığından emin olun.")
        print("3. Privacy settings'ten kameraya izin verildiğini kontrol edin.")
    print("="*40)

if __name__ == "__main__":
    check_cameras()
