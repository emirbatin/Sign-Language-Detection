# model_training.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam

# MacOS M2 optimizasyonları
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # TensorFlow uyarı mesajlarını azalt

# Modülleri içe aktar
from config import Config
from models.action_detection_model import actions, no_sequences, sequence_length, DATA_PATH, create_model

def train_model():
    """Model eğitimini gerçekleştirir"""
    print("=== Model Eğitimi Başlıyor ===")
    
    # Hareketleri etiketlere atama yapacak bir sözlük oluştur
    label_map = {label: num for num, label in enumerate(Config.ACTIONS)}

    sequences, labels = [], []

    # Veri setinden özellikleri ve etiketleri çıkar
    for action in Config.ACTIONS:
        print(f"'{action}' hareketi için veri yükleniyor...")
        for sequence in range(Config.NO_SEQUENCES):
            window = []
            for frame_num in range(Config.SEQUENCE_LENGTH):
                npy_path = os.path.join(Config.DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                if os.path.exists(npy_path):
                    res = np.load(npy_path)
                    window.append(res)
                else:
                    print(f"Uyarı: {npy_path} dosyası bulunamadı, atlanıyor.")
            
            if len(window) == Config.SEQUENCE_LENGTH:
                sequences.append(window)
                labels.append(label_map[action])
            else:
                print(f"Uyarı: {action} eyleminin {sequence} dizisi eksik kare içeriyor, atlanıyor.")

    # Özellikleri ve etiketleri numpy dizilerine dönüştür
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    
    # Veri seti boş mu kontrol et
    if len(X) == 0:
        print("Hata: Veri seti boş, eğitim yapılamıyor.")
        return
    
    print(f"Veri seti yüklendi: {X.shape} özellikler, {y.shape} etiketler")
    
    # Eğitim ve test setlerini oluştur
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TRAIN_TEST_SPLIT)
    
    print(f"Eğitim seti: {X_train.shape}")
    print(f"Test seti: {X_test.shape}")

    # Dizinleri oluştur
    log_dir = os.path.join('Logs')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    # TensorBoard ve Early Stopping geri çağrılarını tanımla
    tb_callback = TensorBoard(log_dir=log_dir)
    early_stopping_callback = EarlyStopping(
        monitor='val_loss', 
        patience=Config.EARLY_STOPPING_PATIENCE, 
        restore_best_weights=True
    )

    # Optimizleyici oluştur
    optimizer = Adam(learning_rate=Config.LEARNING_RATE)

    # Modeli oluştur
    model = create_model()
    
    # Modeli derle
    model.compile(
        optimizer=optimizer, 
        loss='categorical_crossentropy', 
        metrics=['categorical_accuracy']
    )

    # Modeli eğit
    print("\nModel eğitimi başlıyor...")
    model.fit(
        X_train, y_train, 
        epochs=Config.MAX_EPOCHS, 
        batch_size=Config.BATCH_SIZE,
        callbacks=[early_stopping_callback, tb_callback], 
        validation_data=(X_test, y_test)
    )

    # Model özetini görüntüle
    model.summary()

    # Modelin test verisi üzerinde tahminlerini al
    print("\nModel değerlendiriliyor...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test doğruluğu: {test_acc:.4f}")
    
    res = model.predict(X_test)

    # Tahmin edilen etiketleri ve gerçek etiketleri görüntüle
    predicted_labels = [Config.ACTIONS[np.argmax(pred)] for pred in res]
    true_labels = [Config.ACTIONS[np.argmax(true)] for true in y_test]

    # Sadece ilk 10 tahmini göster
    print("\nTahmin Örnekleri (ilk 10):")
    for i, (pred, true) in enumerate(zip(predicted_labels, true_labels)):
        if i < 10:
            print(f"Örnek {i+1}: Tahmin = {pred}, Gerçek = {true}")

    # Eğitilmiş modeli kaydet
    print("\nModel kaydediliyor...")
    model.save(Config.KERAS_MODEL_PATH)
    print(f"Keras modeli '{Config.KERAS_MODEL_PATH}' olarak kaydedildi.")
    
    # TensorFlow Lite modelini oluştur
    try:
        print("TFLite modeli oluşturuluyor...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter._experimental_lower_tensor_list_ops = False
        tflite_model = converter.convert()
        
        # TensorFlow Lite model dosyasını kaydet
        with open(Config.TFLITE_MODEL_PATH, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TensorFlow Lite modeli '{Config.TFLITE_MODEL_PATH}' olarak kaydedildi.")
    except Exception as e:
        print(f"TFLite dönüşümü sırasında hata: {e}")
    
    print("\n=== Model Eğitimi Tamamlandı ===")

if __name__ == "__main__":
    train_model()