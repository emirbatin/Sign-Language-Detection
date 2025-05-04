# save_to_csv.py
import os
import numpy as np
import pandas as pd
from models.action_detection_model import actions, no_sequences, sequence_length, DATA_PATH

def save_to_csv(actions):
    """Eğitim verilerini CSV dosyasına kaydeder"""
    csv_file_path = os.path.join(DATA_PATH, "sign_language_data.csv")
    
    # Verileri toplayacak liste
    all_data = []
    
    # Her hareket için verileri topla
    for action_idx, action in enumerate(actions):
        print(f"'{action}' verileri toplanıyor...")
        
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                # .npy dosya yolu
                npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                
                if os.path.exists(npy_path):
                    # Anahtar noktaları yükle
                    keypoints = np.load(npy_path)
                    
                    # Veri satırı oluştur
                    row_data = {
                        'action': action,
                        'action_id': action_idx,
                        'sequence': sequence,
                        'frame': frame_num
                    }
                    
                    # Anahtar noktaları ekle
                    for i, kp in enumerate(keypoints):
                        row_data[f'keypoint_{i}'] = kp
                    
                    all_data.append(row_data)
    
    # DataFrame oluştur ve CSV'ye kaydet
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(csv_file_path, index=False)
        print(f"Veriler CSV'ye kaydedildi: {csv_file_path}")
        print(f"Toplam {len(all_data)} satır veri kaydedildi.")
        return True
    else:
        print("Kaydedilecek veri bulunamadı!")
        return False