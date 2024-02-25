import sys
sys.path.append("./models/action_recognition_model.py")
import sys
sys.path.append("./models/action_detection_model.py")
import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from action_detection_model import actions, no_sequences, sequence_length, DATA_PATH


# Model eğitimini gerçekleştiren fonksiyon
def train_model():
    # Hareketleri etiketlere atama yapacak bir sözlük oluştur
    label_map = {label:num for num, label in enumerate(actions)}

    sequences, labels = [], []

    # Veri setinden özellikleri ve etiketleri çıkar
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)

            sequences.append(window)
            labels.append(label_map[action])
    
    # Özellikleri ve etiketleri numpy dizilerine dönüştür
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    # Eğitim ve test setlerini oluştur
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    # TensorBoard geri çağrısını tanımla
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Modeli oluştur

    optimizer = Adam(learning_rate=0.001)

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(30, 2172)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(128, return_sequences=False, activation='relu'))
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(actions.shape[0], activation='softmax'))


    # Modeli derle
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # Modeli eğit
    model.fit(X_train, y_train, epochs=1000, callbacks=[early_stopping_callback, tb_callback], validation_data=(X_test, y_test))

    # Model özetini görüntüle
    model.summary()

    # Modelin test verisi üzerinde tahminlerini al
    res = model.predict(X_test)

    # Tahmin edilen etiketleri ve gerçek etiketleri görüntüle
    predicted_labels = [actions[np.argmax(pred)] for pred in res]
    true_labels = [actions[np.argmax(true)] for true in y_test]

    print("Predicted Labels:", predicted_labels)
    print("True Labels:", true_labels)

    # Eğitilmiş modeli kaydet
    model_path = 'ML_Models/action.keras'  # 'action.keras' dosyasını saklamak için yol
    model.save(model_path)
    print("Keras modeli '{}' olarak kaydedildi.".format(model_path))
    
    # TensorFlow Lite modelini oluştur
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    
    # TensorFlow Lite model dosyasını kaydet
    tflite_model_path = 'ML_Models/action.tflite'  # 'action.tflite' dosyasını saklamak için yol
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    
    print("TensorFlow Lite modeli '{}' olarak kaydedildi.".format(tflite_model_path))

    
if __name__ == "__main__":
    # Örnek hareketler, veri yolu ve diğer parametreleri ayarla
    actions = np.array(['konnichiwa', 'arigatou', 'gomen', 'suki', 'nani', 'daijoubu', 'namae', 'genki'])
    DATA_PATH = os.path.join("MP_Data")
    no_sequences = 30
    sequence_length = 30
    # Model eğitimini başlat
    train_model()
