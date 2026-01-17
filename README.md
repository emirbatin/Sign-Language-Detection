# İşaret Dili Tanıma Sistemi

Bu proje, gerçek zamanlı işaret dili tanıma için yapay zeka tabanlı bir sistemdir. MediaPipe ve TensorFlow kullanılarak geliştirilmiştir.

## 🎯 Özellikler

- **Gerçek Zamanlı Tanıma**: Webcam üzerinden canlı işaret dili tanıma
- **Modern UI**: Gradio tabanlı kullanıcı dostu arayüz
- **Jupyter Notebook**: Eğitim sürecini adım adım takip edebilme
- **8 Farklı Hareket**: Japonca işaret dili hareketleri (konnichiwa, arigatou, gomen, suki, nani, daijoubu, namae, genki)
- **Görselleştirme**: Eğitim metrikleri ve confusion matrix

## 📋 Gereksinimler

Python 3.8 veya üzeri gereklidir. Gerekli kütüphaneler:

```bash
pip install -r requirements.txt
```

### Ana Kütüphaneler:
- TensorFlow 2.9+
- OpenCV 4.5+
- MediaPipe 0.8+
- Gradio 4.0+
- Jupyter Notebook

## 🚀 Hızlı Başlangıç

### 1. Jupyter Notebook ile Eğitim

En basit ve düzenli yöntem Jupyter Notebook kullanmaktır:

```bash
jupyter notebook sign_language_training.ipynb
```

Notebook içinde sırayla:
1. **Kütüphaneleri yükleyin** - İlk hücreyi çalıştırın
2. **Veri toplayın** - Webcam ile her hareket için 30 video kaydedin
3. **Modeli eğitin** - LSTM tabanlı derin öğrenme modeli
4. **Sonuçları görselleştirin** - Doğruluk grafikleri, confusion matrix
5. **Modeli kaydedin** - `ML_Models/action.keras` olarak kaydedilir

### 2. Gradio UI ile Test

Model eğitildikten sonra modern web arayüzünü başlatın:

```bash
python app_gradio.py
```

Tarayıcınızda `http://localhost:7860` adresini açın.

**Kullanım:**
- Webcam erişimine izin verin
- El hareketinizi gösterin
- Sistem otomatik olarak tanıyacak
- Güven skorunu ve olasılık dağılımını göreceksiniz

## 📁 Proje Yapısı

### ✨ Minimalist ve Temiz Yapı:

```
Sign-Language-Detection/
├── 📓 sign_language_training.ipynb  # TÜM eğitim kodu burada! ⭐
├── 🎨 app_gradio.py                 # Sadece test UI'ı
├── 📋 requirements.txt              # Gerekli paketler
├── 📖 README.md                     # Bu dosya
└── 📊 MODEL_IMPROVEMENTS.md         # Model detayları

# Eğitim sırasında oluşacaklar:
├── MP_Data/                         # Toplanan veri
└── ML_Models/                       # Eğitilmiş modeller
    ├── action.keras
    └── action.tflite
```

### 🎯 Sadece 2 Dosya!

**1. `sign_language_training.ipynb`**
- ✅ Tüm fonksiyonlar notebook içinde
- ✅ Veri toplama
- ✅ Model tanımları (2 seçenek)
- ✅ Eğitim
- ✅ Görselleştirme
- ✅ Değerlendirme

**2. `app_gradio.py`**
- ✅ Sadece test UI'ı
- ✅ Eğitilmiş modeli yükler
- ✅ Webcam'den tahmin yapar

### 🗑️ Temizlenen Gereksiz Klasörler:

Artık bunlar YOK (silinmiştir):
- ~~models/~~ → Her şey notebook içinde
- ~~utils/~~ → Her şey notebook içinde
- ~~main.py~~ → Notebook kullan
- ~~data_collecting.py~~ → Notebook içinde
- ~~model_training.py~~ → Notebook içinde
- ~~app_testing.py~~ → app_gradio.py kullan
- ~~save_to_csv.py~~ → Notebook içinde
- ~~config.py~~ → Notebook içinde

### 🎨 Yeni Özellikler:

#### 1. **sign_language_training.ipynb**
- ✅ Tüm eğitim süreci tek notebook'ta
- ✅ Adım adım açıklamalar
- ✅ Görselleştirmeler (grafik, confusion matrix)
- ✅ Interaktif çalışma ortamı
- ✅ Her hücre bağımsız çalışabilir

#### 2. **app_gradio.py**
- ✅ Modern web tabanlı UI
- ✅ Gerçek zamanlı tahmin
- ✅ Güven skoru gösterimi
- ✅ Olasılık dağılımı grafiği
- ✅ Responsive tasarım
- ✅ Kullanımı kolay arayüz

## 🎓 Kullanım Kılavuzu

### Veri Toplama

Notebook içinde veri toplama bölümünü çalıştırın:
- Her hareket için 30 video kaydedilir
- Her video 30 kare içerir
- Webcam üzerinden gerçek zamanlı kayıt
- ENTER ile başlat, Q ile çık

### Model Eğitimi

Notebook otomatik olarak:
- Veriyi yükler ve hazırlar
- Train/Test setlerine ayırır
- LSTM + Conv1D hibrit model oluşturur
- Early stopping ile eğitir
- En iyi modeli kaydeder

### Model Mimarisi

#### 🎯 İki Model Seçeneği:

**1. Geliştirilmiş Model (create_model)** - Önerilen
```
Conv1D(64, k=5) → BatchNorm → Dropout(0.2)
    ↓
Conv1D(128, k=3) → BatchNorm → MaxPool → Dropout(0.2)
    ↓
Conv1D(256, k=3) → BatchNorm → MaxPool → Dropout(0.3)
    ↓
Bidirectional LSTM(128) → LayerNorm → Dropout(0.3)
    ↓
Bidirectional LSTM(64) → LayerNorm → Dropout(0.4)
    ↓
Dense(128) → BatchNorm → Dropout(0.4)
    ↓
Dense(64) → Dropout(0.3)
    ↓
Dense(8, softmax) [Çıkış]
```

**2. Advanced Model (create_model_advanced)** - En Güçlü
```
Conv1D Blocks (64→128→256) + Residual Connections
    ↓
Bidirectional LSTM(128) → LayerNorm
    ↓
Bidirectional LSTM(64) → LayerNorm
    ↓
Multi-Head Attention (4 heads) + Residual
    ↓
GlobalAveragePooling1D
    ↓
Dense(128) → Dense(64) → Dense(8, softmax)
```

**Özellikler:**
- ✅ **Bidirectional LSTM**: İleri ve geri yönlü zamansal öğrenme
- ✅ **Multi-Head Attention**: Önemli frame'lere odaklanma (advanced)
- ✅ **Residual Connections**: Derin ağda gradient akışını iyileştirme (advanced)
- ✅ **Layer Normalization**: Eğitim stabilitesi
- ✅ **L2 Regularization**: Overfitting önleme
- ✅ **Dropout**: Genelleme yeteneği

## 🔧 Özelleştirme

### Kendi İşaret Dilinizi Eklemek

1. **Notebook'ta Config sınıfını düzenleyin:**

```python
class Config:
    ACTIONS = np.array(['merhaba', 'teşekkürler', 'güle güle'])  # Kendi hareketleriniz
    NO_SEQUENCES = 30
    SEQUENCE_LENGTH = 30
```

2. **Veri toplama hücresini çalıştırın**
3. **Modeli yeniden eğitin**

### Parametreleri Ayarlama

```python
class Config:
    BATCH_SIZE = 32              # Batch boyutu
    LEARNING_RATE = 0.001        # Öğrenme oranı
    MAX_EPOCHS = 1000            # Maksimum epoch
    EARLY_STOPPING_PATIENCE = 15 # Sabır süresi
    PREDICTION_THRESHOLD = 0.5   # Tahmin eşiği
```

## 🚀 Model Performansı ve İyileştirmeler

### 📊 Model Karşılaştırması:

| Model | Parametreler | Doğruluk (tahm.) | Eğitim Süresi | Özellikler |
|-------|-------------|------------------|---------------|------------|
| **Eski Model** | ~500K | %75-80 | 30-40 dk | Basic LSTM + Conv1D |
| **Geliştirilmiş** | ~1.2M | %85-90 | 40-50 dk | Bi-LSTM + LayerNorm + Dropout |
| **Advanced** | ~2.5M | %90-95 | 60-80 dk | + Attention + Residual |

### 🎯 Hangi Modeli Seçmeli?

**Başlangıç için:**
```python
model = create_model()  # Geliştirilmiş model
```
- ✅ Hızlı eğitim
- ✅ İyi performans
- ✅ Daha az kaynak kullanımı

**En iyi performans için:**
```python
model = create_model_advanced()  # Advanced model
```
- ✅ En yüksek doğruluk
- ✅ Attention mekanizması
- ✅ Residual connections
- ⚠️ Daha uzun eğitim
- ⚠️ Daha fazla GPU belleği

### 💡 Model İyileştirme İpuçları

1. **Veri Artırma**: Daha fazla veri toplayın (NO_SEQUENCES = 50)
2. **Learning Rate**: ReduceLROnPlateau otomatik ayarlıyor
3. **Batch Size**: GPU belleğinize göre 32-64 arası
4. **Epochs**: Early stopping otomatik duruyor
5. **Data Augmentation**: Keypoint'lere gürültü ekleyin

## 🐛 Sorun Giderme

### Model Yüklenmiyor
```bash
# Notebook'ta model eğitim hücrelerini çalıştırdığınızdan emin olun
# ML_Models/action.keras dosyası oluşmalı
```

### Webcam Açılmıyor
```python
# Config içinde kamera index'ini değiştirin
CAMERA_INDEX = 0  # veya 1, 2, etc.
```

### Gradio Hatası
```bash
# Gradio'yu güncelleyin
pip install --upgrade gradio
```

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 👨‍💻 Geliştirici

**Emir Batın**

- GitHub: [@emirbatin](https://github.com/emirbatin)

## 🙏 Teşekkürler

- MediaPipe ekibine el/yüz/poz algılama için
- TensorFlow ekibine derin öğrenme framework'ü için
- Gradio ekibine modern UI framework'ü için

---

## 🆚 Eski vs Yeni Kullanım

### ❌ Eski Yöntem (Karmaşık):
```bash
python main.py  # Tkinter arayüzü açılır
# Veri toplama butonuna tıkla
# Model eğitimi butonuna tıkla
# Test uygulaması butonuna tıkla
```

### ✅ Yeni Yöntem (Düzenli):
```bash
# 1. Eğitim için
jupyter notebook sign_language_training.ipynb
# Hücreleri sırayla çalıştır

# 2. Test için
python app_gradio.py
# Tarayıcıda aç: http://localhost:7860
```

## 📸 Ekran Görüntüleri

### Jupyter Notebook
- Adım adım eğitim süreci
- İnteraktif grafikler
- Kod açıklamaları

### Gradio UI
- Modern web arayüzü
- Gerçek zamanlı tahmin
- Güven skorları
- Olasılık grafikleri

---
