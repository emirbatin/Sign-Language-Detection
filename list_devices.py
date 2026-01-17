
# Bu script, sistemdeki kameraların İSİMLERİNİ listeler.
# Çalıştırmadan önce: pip install pygrabber
from pygrabber.dshow_graph import FilterGraph

def list_devices():
    try:
        print("📡 DirectShow üzerinden cihazlar taranıyor...")
        graph = FilterGraph()
        devices = graph.get_input_devices()
        
        if not devices:
            print("❌ Hiçbir video giriş cihazı bulunamadı!")
            return

        print(f"\n✅ {len(devices)} adet cihaz bulundu:")
        print("="*40)
        for i, device_name in enumerate(devices):
            print(f"Index {i}: {device_name}")
        print("="*40)
        print("\nEğer 'Camo Studio' listedeyse, yanındaki Index numarasını")
        print("collect_holistic_data.py dosyasında cv2.VideoCapture(INDEX) içine yazın.")

    except Exception as e:
        print(f"Hata: {e}")
        print("Lütfen 'pip install pygrabber' komutunu çalıştırdığınızdan emin olun.")

if __name__ == "__main__":
    list_devices()
