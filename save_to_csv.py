import sys
sys.path.append("./models/action_detection_model.py")

import os
import csv
import numpy as np
from action_detection_model import actions, no_sequences, sequence_length, DATA_PATH
# Diğer fonksiyonları içe aktarabilirsiniz

# Verileri CSV dosyasına kaydetme fonksiyonu
def save_to_csv(actions):
    Folder = "Dataset"  # CSV dosyasının kaydedileceği klasör
    csv_file_path = os.path.join(Folder, 'dataset.csv')
    with open(csv_file_path, mode='w', newline='') as file:
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
    
    print("CSV başarıyla kaydedildi.")

if __name__ == "__main__":
    actions = np.array(['hello', 'thanks', 'howareyou'])
    DATA_PATH = os.path.join("MP_Data")
    no_sequences = 30
    sequence_length = 30
    save_to_csv(actions)
