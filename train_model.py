
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import os
import time

# Constants
DATA_PATH = os.path.join('MP_Data') 
actions = np.array(['konnichiwa', 'arigatou', 'gomen'])
no_sequences = 30
sequence_length = 30

label_map = {label:num for num, label in enumerate(actions)}

print("Veri yükleniyor...")

sequences, labels = [], []

try:
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
except Exception as e:
    print(f"Veri yüklerken hata oluştu: {e}")
    print("Lütfen önce 'collect_holistic_data.py' ile veri toplayın!")
    exit()

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

print(f"Eğitim veri seti boyutu: {X.shape}")
print(f"Input shape: {X.shape[1:]}") # (30, 2172) olmalı

# Model Mimarisi
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 2172)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print("Model özeti:")
model.summary()

print("Eğitim başlıyor...")
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])

# Modeli kaydet (H5 formatı - User request)
model_path = 'action.h5'
model.save(model_path, save_format='h5')
print(f"Model başarıyla kaydedildi: {model_path}")

# Alternatif olarak ml-server klasörüne de kopyalanabilir
server_model_path = os.path.join('ml-server', 'models', 'action.h5')
if os.path.exists(os.path.dirname(server_model_path)):
    model.save(server_model_path, save_format='h5')
    print(f"Model sunucu klasörüne de kopyalandı: {server_model_path}")

# Test
res = model.predict(X_test)
print("Test tahmin örneği:")
print(actions[np.argmax(res[0])])
print(actions[np.argmax(y_test[0])])
