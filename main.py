import sys
sys.path.append("models")  # Modül dizinini ekleyin

import tkinter as tk
from data_collecting import collecting_data
from model_training import train_model
from app_testing import test_app
from save_to_csv import save_to_csv
from action_detection_model import actions, model as model

# Ana uygulama penceresini oluştur
app = tk.Tk()
app.title("Machine Learning Uygulaması")

# "Collecting Data" butonunu oluştur ve pencereye ekle
collecting_data_button = tk.Button(app, text="Collecting Data", command=collecting_data)
collecting_data_button.pack(pady=10)

# "Train Model" butonunu oluştur ve pencereye ekle
train_model_button = tk.Button(app, text="Train Model", command=train_model)
train_model_button.pack(pady=10)

# "Test App" butonunu oluştur ve pencereye ekle
test_app_button = tk.Button(app, text="Test App", command=lambda: test_app(model, actions))
test_app_button.pack(pady=10)

# "Save to CSV" butonunu oluştur ve pencereye ekle
save_to_csv_button = tk.Button(app, text="Save to CSV", command=lambda: save_to_csv(actions))
save_to_csv_button.pack(pady=10)

# Uygulamayı başlat
app.mainloop()
