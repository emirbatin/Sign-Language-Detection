# 🍎 MacOS Kurulum Rehberi

Bu projeyi Mac üzerinde çalıştırmak için aşağıdaki adımları izleyin.

## 1. Kodları Aktarma
Bu bilgisayardaki (`collect_holistic_data.py`, `train_model.py`, `app_gradio.py` vb.) dosyaların Mac'e geçtiğinden emin olun. (Git push/pull veya AirDrop ile).

## 2. Terminal & Kurulum
Mac terminalini açın ve proje klasörüne gidin.

```bash
# Sanal ortam oluştur
python3 -m venv venv

# Aktif et
source venv/bin/activate

# Kütüphaneleri yükle
# (Apple Silicon M1/M2/M3 kullanıyorsanız tensorflow otomatik uyumlu sürümü kuracaktır)
pip install -r requirements.txt
```

> **Önemli Not (Apple Silicon):** Eğer `hdf5` veya `tensorflow` hatası alırsanız:
> ```bash
> pip install tensorflow-macos tensorflow-metal
> ```

## 3. Çalıştırma Sırası

### A. Veri Toplama
Kameranızın iznini verin.
```bash
python collect_holistic_data.py
```
*Not: Eğer kamera açılmazsa `python check_cameras.py` dosyasını orada da çalıştırıp Index numarasını kontrol edin.*

### B. Eğitim
```bash
python train_model.py
```

### C. Test
```bash
python test_realtime.py
# veya
python app_gradio.py
```
