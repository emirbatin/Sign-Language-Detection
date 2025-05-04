# main.py
import sys
sys.path.append("./models")  # Modül dizinlerini ekleyin
sys.path.append("./utils")

import os
import tkinter as tk
from tkinter import ttk, messagebox
import threading

# MacOS M2 optimizasyonları
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # TensorFlow uyarı mesajlarını azalt

# İçe aktarmalar
from config import Config
from data_collecting import collecting_data
from model_training import train_model
from app_testing import test_app
from save_to_csv import save_to_csv
from models.action_detection_model import actions, model

def run_task(app, func, status_var, success_message):
    """Görevi ana thread'de çalıştırır ve durum mesajını günceller"""
    status_var.set("İşlem çalıştırılıyor...")
    app.update_idletasks()
    
    try:
        # MacOS M2'de thread kullanmadan ana thread'de çalıştır
        func()
        status_var.set(success_message)
    except Exception as e:
        status_var.set(f"Hata: {str(e)}")
        messagebox.showerror("Hata", str(e))

def main():
    # Gerekli dizinleri oluştur
    os.makedirs(Config.DATA_PATH, exist_ok=True)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    # Model durumunu kontrol et
    model_status = "Model yüklendi ✓" if model is not None else "Model henüz eğitilmedi ✗"
    
    # Ana uygulama penceresini oluştur
    app = tk.Tk()
    app.title("İşaret Dili Tanıma")
    app.geometry("600x500")
    app.configure(bg="#1E2952")  # Koyu lacivert arka plan
    
    # TTK Stil ayarları
    style = ttk.Style()
    style.theme_use("clam")  # Tema belirleme
    
    # Durum değişkeni
    status_var = tk.StringVar()
    status_var.set("Hazır")
    
    # Ana çerçeve
    main_frame = tk.Frame(app, bg="#1E2952", padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Başlık
    title_label = tk.Label(main_frame, 
                           text="İŞARET DİLİ TANIMA SİSTEMİ", 
                           font=("Arial", 20, "bold"), 
                           fg="#FFFFFF", 
                           bg="#1E2952")
    title_label.pack(pady=(0, 5))
    
    # Alt başlık
    subtitle_label = tk.Label(main_frame,  
                              font=("Arial", 12, "italic"), 
                              fg="#4C9EFD", 
                              bg="#1E2952")
    subtitle_label.pack(pady=(0, 20))
    
    # Model durum kartı
    status_frame = tk.Frame(main_frame, bg="#2A3563", bd=1, relief=tk.RAISED, padx=10, pady=10)
    status_frame.pack(fill=tk.X, pady=(0, 20))
    
    status_title = tk.Label(status_frame, 
                           text="MODEL DURUMU", 
                           font=("Arial", 10, "bold"), 
                           fg="#AAAAAA", 
                           bg="#2A3563")
    status_title.pack(anchor=tk.W)
    
    # İndikatör rengi
    indicator_color = "#4CAF50" if model is not None else "#FFA726"  # Yeşil veya turuncu
    
    status_content = tk.Frame(status_frame, bg="#2A3563")
    status_content.pack(fill=tk.X, pady=(5, 0))
    
    indicator = tk.Canvas(status_content, width=15, height=15, bg="#2A3563", highlightthickness=0)
    indicator.create_oval(2, 2, 13, 13, fill=indicator_color, outline="")
    indicator.pack(side=tk.LEFT, padx=(0, 5))
    
    status_label = tk.Label(status_content, 
                           text=model_status, 
                           font=("Arial", 10), 
                           fg="#FFFFFF", 
                           bg="#2A3563")
    status_label.pack(side=tk.LEFT)
    
    # Butonlar için stil
    button_style = {
        "font": ("Arial", 12, "bold"),
        "fg": "white",
        "activeforeground": "white",
        "bd": 0,
        "highlightthickness": 0,
        "padx": 20,
        "pady": 10,
        "width": 20
    }
    
    # Butonlar çerçevesi
    buttons_frame = tk.Frame(main_frame, bg="#1E2952")
    buttons_frame.pack(fill=tk.BOTH, expand=True)
    
    # Veri Toplama butonu - MacOS M2 uyumlu: doğrudan çağrı
    collect_btn = tk.Button(
        buttons_frame, 
        text="Veri Toplama", 
        bg="#4169E1",  # Royal Blue
        activebackground="#3A5FCD",
        command=lambda: app.after(100, lambda: run_task(
            app, collecting_data, status_var, "Veri toplama tamamlandı"
        )),
        **button_style
    )
    collect_btn.pack(pady=10, fill=tk.X)
    
    # Model Eğitimi butonu - MacOS M2 uyumlu: doğrudan çağrı
    train_btn = tk.Button(
        buttons_frame, 
        text="Model Eğitimi", 
        bg="#4169E1",
        activebackground="#3A5FCD",
        command=lambda: app.after(100, lambda: run_task(
            app, train_model, status_var, "Model eğitimi tamamlandı"
        )),
        **button_style
    )
    train_btn.pack(pady=10, fill=tk.X)
    
    # Test Uygulaması butonu - MacOS M2 uyumlu: doğrudan çağrı
    test_btn = tk.Button(
        buttons_frame, 
        text="Uygulamayı Test Et", 
        bg="#4169E1",
        activebackground="#3A5FCD",
        command=lambda: app.after(100, lambda: run_task(
            app, lambda: test_app(model, actions), status_var, "Test uygulaması çalıştırıldı"
        )),
        **button_style
    )
    test_btn.pack(pady=10, fill=tk.X)
    
    # CSV'ye Kaydet butonu - MacOS M2 uyumlu: doğrudan çağrı
    csv_btn = tk.Button(
        buttons_frame, 
        text="CSV'ye Kaydet", 
        bg="#4169E1",
        activebackground="#3A5FCD",
        command=lambda: app.after(100, lambda: run_task(
            app, lambda: save_to_csv(actions), status_var, "CSV kaydedildi"
        )),
        **button_style
    )
    csv_btn.pack(pady=10, fill=tk.X)
    
    # Çıkış butonu
    exit_btn = tk.Button(
        buttons_frame, 
        text="Çıkış", 
        bg="#E53935",  # Kırmızı
        activebackground="#C62828",
        command=app.destroy,
        **button_style
    )
    exit_btn.pack(pady=10, fill=tk.X)
    
    # Durum çubuğu
    status_bar = tk.Frame(main_frame, bg="#2A3563", height=30)
    status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(20, 0))
    
    status_message = tk.Label(
        status_bar, 
        textvariable=status_var, 
        font=("Arial", 9), 
        fg="#FFFFFF", 
        bg="#2A3563",
        anchor=tk.W,
        padx=10,
        pady=5
    )
    status_message.pack(fill=tk.X)
    
    # Copyright
    copyright_label = tk.Label(
        app, 
        text="© 2025 İşaret AI - Tüm Hakları Saklıdır", 
        font=("Arial", 8), 
        fg="#AAAAAA", 
        bg="#1E2952",
        pady=5
    )
    copyright_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    # MacOS M2 uyumluluğu açıklaması
    messagebox.showinfo(
        "MacOS M2 Uyumluluğu", 
        "Bu uygulama, MacOS M2 işlemcilerde çalışacak şekilde optimize edilmiştir. "
        "Bazı işlemler biraz yavaş olabilir, ancak daha güvenilir çalışacaktır."
    )
    
    # Uygulamayı başlat
    app.mainloop()

if __name__ == "__main__":
    main()