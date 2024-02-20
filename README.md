<h1>English</h1>
This project is trained using a neural network algorithm with the overall goal of converting the user's hand signs into text. At the moment, only 3 words have been trained on Japanese sign language and it works with an average accuracy. New languages can be added with future improvements.

Before run, install these libraries:

- pip install numpy
- pip install tensorflow
- pip install tkinter
- pip install mediapipe
- pip install opencv-python

Currently, the project is trained on 3 words in Japanese sign language, and in the future the performance and number of words will be improved and increased.

If you want to teach the sign language of your country, delete the folders in MP_Data and change the words in the actions array in the action_detection_model.py file in the models folder or add new words

Then from the main.py file, start the project and train the model on the screen that opens.

<h1>日本語</h1>

<p>このプロジェクトは、ユーザーの手話をテキストに変換することを全体的な目標として、ニューラルネットワーク・アルゴリズムを使って訓練されます｡ 現在、日本語の手話は3語のみ学習されており、平均的な精度で動作します。今後の改良により、新しい言語が追加される予定です。</p>

ランする前に、以下のライブラリをインストールしてください。

- pip install numpy
- pip install tensorflow
- pip install tkinter
- pip install mediapipe
- pip install opencv-python

現在、プロジェクトは日本の手話で3つの単語を学習しています。将来的には、性能や単語数を改善および増やす予定です。

あなたの国の手話を教えたい場合は、MP_Dataフォルダ内のフォルダを削除し、modelsフォルダ内のaction_detection_model.pyファイルのactions配列内の単語を変更するか、新しい単語を追加してください。

次に、main.pyファイルからプロジェクトを起動し、開かれる画面でモデルを訓練してください。

<h1>Türkçe</h1>
Bu proje, kullanıcının el işaretlerini metne dönüştürme amacı ile bir sinir ağı algoritması kullanılarak eğitilmiştir. Şu anda Japon işaret dili üzerinde sadece 3 kelime eğitilmiştir ve ortalama bir doğrulukla çalışmaktadır. Gelecekteki geliştirmelerle yeni diller eklenebilir.

Çalıştırmadan önce bu kütüphaneleri yükleyin:

- pip numpy yükleyin
- pip install tensorflow
- pip install tkinter
- pip install mediapipe
- pip install opencv-python

Şu anda proje Japon işaret dilinde 3 kelime üzerinde eğitilmiştir ve gelecekte performans ve kelime sayısı geliştirilerek arttırılacaktır.

Eğer ülkenizin işaret dilini öğretmek istiyorsanız MP_Data içindeki klasörleri silin ve models klasöründeki action_detection_model.py dosyasındaki actions dizisindeki kelimeleri değiştirin veya yeni kelimeler ekleyin

Daha sonra main.py dosyasından projeyi başlatın ve açılan ekranda modeli eğitin.
