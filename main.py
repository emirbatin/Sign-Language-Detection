import os
import cv2
import csv
import numpy as np
import tkinter as tk
from tkinter import messagebox
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf

# MediaPipe Holistic modelini ayarla
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Çizim zart zurtları

#Verilerin çıkarılacağı yol, numpys arrays
DATA_PATH = os.path.join("MP_Data")

#Harketlerimizi tespit etmeye çalışacak
actions = np.array(['hello','thanks','howareyou'])

#30 Video değerinde verilerimiz olacak
no_sequences = 30

#Video 30 kare uzunluğunda olacak 
sequence_length = 30

# Eğer action.keras dosyası mevcut değilse model değişkenini None olarak ayarla
model_file_path = 'action.keras'
model = load_model(model_file_path) if os.path.exists(model_file_path) else None

# Şimdi model değişkeni None değerine sahip olacak, çünkü dosya bulunmuyor
print(model)


# Fonksiyonları tanımla
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR'dan RGB'ye dönüştür
    image.flags.writeable = False                   # Görüntüyü salt okunur yap
    results = model.process(image)                  # Modelle algılama yap
    image.flags.writeable = True                    # Görüntüyü tekrar yazılabilir yap
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # RGB'den BGR'ye dönüştür
    return image, results

#Default Landmarks
def draw_landmarks(image, results):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp.solutions.holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
        
#Custom Landmarks (!)        
def draw_styled_landmarks(image, results):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Yüz bağlantılarını çiz
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            results.face_landmarks, 
            mp_holistic.FACEMESH_TESSELATION,  # Yüz landmarklarının bağlantıları için
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
        )
        
    # Vücut pozisyon bağlantılarını çiz
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            mp_holistic.POSE_CONNECTIONS,  # Vücut pozisyonlarının bağlantıları için
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
        )
        
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            results.left_hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS, # Sol el pozisyonlarının bağlantıları için
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=2, circle_radius=1),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
        )
        
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS, # Sağ el pozisyonlarının bağlantıları için
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=2, circle_radius=1),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
        )


#mp_holistic.POSE_CONNECTIONS

#mp_drawing.draw_landmarks??


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z, res.visibility] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 4)
    lh = np.array([[res.x, res.y, res.z, res.visibility] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 4)
    rh = np.array([[res.x, res.y, res.z, res.visibility] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 4)
    # Anahtar noktaları birleştirin ve istenen şekli elde etmek için tamponlayın veya kesin
    keypoints = np.concatenate([pose, face, lh, rh])
    if len(keypoints) < 2172:
        # Uzunluk 2172'den azsa sıfırlarla doldur
        keypoints = np.pad(keypoints, (0, 2172 - len(keypoints)))
    elif len(keypoints) > 2172:
        # Uzunluk 2172'den büyükse kes
        keypoints = keypoints[:2172]

    return keypoints

#
#
#
####### VERİLERİN TOPLANMASI ##########
#
#
#

def create_directories(actions):
    DATA_PATH = os.path.join("MP_Data")
    no_sequences = 5

    for action in actions:
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

def collecting_data(actions):

    # Veri toplamak için dizinleri oluştur
    create_directories(actions)

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        sequences = []
        labels = []

        for action in actions:
            status_text = f"Press space to start collecting frames for {action}."
            print(status_text)
            messagebox.showinfo("Collecting Data", status_text)

            while True:
                ret, frame = cap.read()
                cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', frame)
                if cv2.waitKey(1) & 0xFF == ord(' '):
                    break

            for sequence in range(no_sequences):
                window = []
                for frame_num in range(30):
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    image, results = mediapipe_detection(frame, holistic)
                    draw_landmarks(image, results)

                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (50, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, f'Collecting frames for {action} Video number {sequence + 1}', (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(image, f'Collecting frames for {action} Video number {sequence + 1}', (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                    keypoints = extract_keypoints(results)

                    #Eğer frame büyüklüğü 2172'den büyük olursa hata verir ve verisetine veriler kaydedilmez
                    if np.shape(keypoints) != (2172,):
                        print(f"Uyumsuz çerçeve boyutu: {np.shape(keypoints)}")
                        continue

                    npy_path = os.path.join("MP_Data", action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    window.append(keypoints)
                    cv2.imshow('OpenCV Feed', image)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

                sequences.append(window)
                labels.append(action)

            print(f"Recording for {action} completed. Press space to start collecting frames for the next word.")

            while True:
                ret, frame = cap.read()
                cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', frame)
                if cv2.waitKey(1) & 0xFF == ord(' '):
                    break

        cap.release()
        cv2.destroyAllWindows()

#
#
#
####### MODELIN EGITILMESI ##########
#
#
#

def train_model(actions):
    label_map = {label:num for num, label in enumerate(actions)}

    label_map

    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)

            sequences.append(window)
            labels.append(label_map[action])
    
    for i, seq in enumerate(sequences[:5]):
        print(f"Dizi {i}:")
        for frame in seq:
            print("Çerçeve Boyutu:", np.shape(frame))

    np.array(sequences).shape

    np.array(labels).shape

    X = np.array(sequences)

    y = to_categorical(labels).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    X.shape

    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 2172)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

# Test kodları (Önemsiz)
    actions.shape[0]

    res = [.7, 0.2, 0.1]

    actions[np.argmax(res)]

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

    model.summary()

    res = model.predict(X_test)

    for i in range(len(res)):
        print(f"Toplam Olasılık {i+1}: {np.sum(res[i])}")

#    actions[np.argmax(res[1])]

#    actions[np.argmax(y_test[1])]


    predicted_labels = [actions[np.argmax(pred)] for pred in res]
    true_labels = [actions[np.argmax(true)] for true in y_test]

    print("Predicted Labels:", predicted_labels)
    print("True Labels:", true_labels)

    model.save('action.keras')
    model.load_weights('action.keras')

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open('action.tflite', 'wb') as f:
        f.write(tflite_model)

    print("TensorFlow Lite modeli 'action.tflite' olarak kaydedildi.")
    

    from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

    yhat = model.predict(X_test)

    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()

    confusion_matrix = multilabel_confusion_matrix(ytrue, yhat)
    accuracy = accuracy_score(ytrue, yhat)

    print("Confusion Matrix:")
    print(confusion_matrix)
    print("Accuracy:", accuracy)


#
#
#
####### UYGULAMANIN TEST EDILMESI ##########
#
#
#


def test_app(actions, model):
    if model is None:
        # Model dosyasının varlığını kontrol et
        if not os.path.exists(model_file_path):
            messagebox.showerror("Error", "Model file not found. Please train the model first.")
            return

        # Model dosyası varsa yükle
        model = load_model(model_file_path)
        print("Model loaded successfully.")
    
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.4

    # MediaPipe Holistic modelini ayarla
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # Algıla
            image, results = mediapipe_detection(frame, holistic)

            # Özellik noktalarını çiz
            draw_landmarks(image, results)

            # 2. Prediction Logic
            if "left_hand_landmarks" in results and "right_hand_landmarks" in results:
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]

                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
                    predictions.append(np.argmax(res))
            else:
                # If either left or right hand landmarks are not detected
                predictions.append(len(actions) - 1)  # Index for "No Sign"

            # 3. Viz Logic
            if len(predictions) > 0 and len(res) > 0 and np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)].any() > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[:1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 1:
                sentence = sentence[-1:]

            cv2.rectangle(image, (0, 0), (500, 80), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 2,
                        cv2.LINE_AA)

            # Ekranı Göster
            cv2.imshow('OpenCV Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

#
#
#
####### CSV OLARAK KAYDETME ##########
#        
#
#

# Verileri CSV dosyasına kaydetme fonksiyonu
def save_to_csv(actions):
    with open('dataset.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        headers = ["Label"] + [f"Keypoint_{i}" for i in range(2172)]  # 2172, her bir çerçevedeki toplam özellik sayısı
        writer.writerow(headers)

        for action in actions:
            for sequence in range(no_sequences):
                for frame_num in range(sequence_length):
                    file_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                    keypoints = np.load(file_path)
                    row = [action] + keypoints.tolist()
                    writer.writerow(row)

#
#
#
####### PENCERE OLUSTURMA ##########
#        
#
#

# Ana uygulama penceresini oluştur
app = tk.Tk()
app.title("Machine Learning Uygulaması")

# "Collecting Data" butonunu oluştur ve pencereye ekle
collecting_data_button = tk.Button(app, text="Collecting Data", command=lambda: collecting_data(actions))
collecting_data_button.pack(pady=10)


# "Train Model" butonunu oluştur ve pencereye ekle
train_model_button = tk.Button(app, text="Train Model", command=lambda: train_model(actions))
train_model_button.pack(pady=10)

# "Test App" butonunu oluştur ve pencereye ekle
test_app_button = tk.Button(app, text="Test App", command=lambda: test_app(actions, model))
test_app_button.pack(pady=10)

# "Save to CSV" butonunu oluştur ve pencereye ekle
save_to_csv_button = tk.Button(app, text="Save to CSV", command=lambda: save_to_csv(actions))
save_to_csv_button.pack(pady=10)

# Uygulamayı başlat
app.mainloop()
