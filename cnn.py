#orjinal dataset 70-30
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, accuracy_score
from google.colab import drive
from keras.utils import plot_model

# Google Drive'ı bağlama
try:
    drive.mount('/content/drive')
except Exception as e:
    print(f"Google Drive bağlanamadı: {e}")
    raise

data_src = '/content/drive/MyDrive/OrjinalDataset'
drive_path = '/content/drive/MyDrive/cnn-orijinal30Split_plot.png'

data = []
y = []

# Veriyi yükleme ve işleme
if not os.path.exists(data_src):
    raise FileNotFoundError(f"Belirtilen veri yolu bulunamadı: {data_src}")

for d in os.listdir(data_src):
    folder_path = os.path.join(data_src, d)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            try:
                img = Image.open(file_path).convert('L')
                img = img.resize((224, 224))
                data.append(np.array(img))
                y.append(d.strip())
            except Exception as e:
                print(f"Hata oluştu: {file_path}, {e}")

# Görüntüleri normalize etme
X = np.array(data) / 255.0
X = X.reshape(X.shape[0], 224, 224, 1)

# Sınıf etiketleri
label_map = {
    "NonDemented": 0,
    "VeryMildDemented": 1,
    "MildDemented": 2,
    "ModerateDemented": 3
}

y_num = [label_map.get(label, -1) for label in y]
if -1 in y_num:
    raise ValueError("Geçersiz etiketler bulundu!")

y = to_categorical(y_num, num_classes=4)

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

# Farklı optimizasyon algoritmalarını test etme
optimizers = ["adam", "rmsprop", "sgd", "adagrad", "adadelta", "nadam"]
neurons = [(32, 32), (64, 64), (32, 32, 32, 32), (32, 32, 64, 64), (64, 64, 64, 64)]

results = {}

for opt in optimizers:
    for neuron in neurons:
        print(f"\nModel: {opt} optimizer, {neuron} nöron")

        cnn = Sequential()
        cnn.add(Conv2D(neuron[0], (3,3), padding="same", activation='relu', input_shape=(224, 224, 1)))
        cnn.add(MaxPooling2D())
        cnn.add(Conv2D(neuron[1], (3,3), padding="same", activation='relu'))
        cnn.add(MaxPooling2D())
        cnn.add(Dropout(0.2))

        if len(neuron) > 2:
            cnn.add(Conv2D(neuron[2], (3,3), padding="same", activation='relu'))
            cnn.add(MaxPooling2D())
            cnn.add(Dropout(0.25))

        if len(neuron) > 3:
            cnn.add(Conv2D(neuron[3], (3,3), padding="same", activation='relu'))
            cnn.add(MaxPooling2D())
            cnn.add(Dropout(0.3))

        cnn.add(Flatten())
        cnn.add(Dense(100, activation='relu'))
        cnn.add(Dense(50, activation='relu'))
        cnn.add(Dense(4, activation='softmax'))

        cnn.compile(optimizer=opt.lower(), loss='categorical_crossentropy', metrics=['accuracy'])

        # Early stopping ve learning rate reduction ekleme
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

        plot_model(cnn, to_file='cnn_{nöron}.png', show_shapes=True, show_layer_names=True)

        # Modeli eğitme
        history = cnn.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])

        # Test seti üzerindeki tahminler
        y_pred = cnn.predict(X_test)
        y_val = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        # Performans metriklerini hesapla
        val_acc = history.history['val_accuracy'][-1]
        test_acc = accuracy_score(y_true, y_val)
        f1 = f1_score(y_true, y_val, average='weighted')
        recall = recall_score(y_true, y_val, average='weighted')

        # Sınıflandırma raporu
        report = classification_report(y_true, y_val, target_names=["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"])

        # Sonuçları sakla
        results[f"{opt}_{neuron}"] = {
            "val_acc": val_acc,
            "test_acc": test_acc,
            "f1_score": f1,
            "recall": recall,
            "classification_report": report,
            "history": history.history
        }

        print(f"✅ Test Accuracy: {test_acc:.4f}")
        print(f"✅ F1 Score (Weighted Average): {f1:.4f}")
        print(f"✅ Recall Score (Weighted Average): {recall:.4f}")
        print(f"✅ Classification Report:\n{report}")

        # Karışıklık Matrisi
        cm = confusion_matrix(y_true, y_val)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'{opt} - Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(4)
        plt.xticks(tick_marks, label_map.keys(), rotation=45)
        plt.yticks(tick_marks, label_map.keys())
        for i in range(4):
            for j in range(4):
                plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

# Sonuç tablosu
results_df = pd.DataFrame({
    'Model': [key for key in results.keys()],
    'Validation Accuracy': [value['val_acc'] for value in results.values()],
    'Test Accuracy': [value['test_acc'] for value in results.values()],
    'F1 Score': [value['f1_score'] for value in results.values()],
    'Recall': [value['recall'] for value in results.values()]
})
print(results_df.sort_values(by='Test Accuracy', ascending=False))

# Tüm modellerin sonuçlarını yazdırma ve grafiklerini çizdirme
print("\nTüm Modellerin Sonuçları:")
for key, value in results.items():
    print(f"\n{key}:")
    print(f"  Validation Accuracy: {value['val_acc']:.4f}")
    print(f"  Test Accuracy: {value['test_acc']:.4f}")
    print(f"  F1 Score: {value['f1_score']:.4f}")
    print(f"  Recall: {value['recall']:.4f}")
    print(f"  Sınıflandırma Raporu:\n{value['classification_report']}")

    # Grafikleri çizdir
    history_data = value["history"]
    plt.figure(figsize=(12, 4))

    # Doğruluk grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history_data['accuracy'], label="Eğitim Doğruluğu")
    plt.plot(history_data['val_accuracy'], label="Validation Doğruluğu")
    plt.title(f"Eğitim ve Validation Doğruluğu ({key})")
    plt.xlabel("Epochs")
    plt.ylabel("Doğruluk")
    plt.legend()

    # Loss grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history_data['loss'], label="Eğitim Loss")
    plt.plot(history_data['val_loss'], label="Validation Loss")
    plt.title(f"Eğitim ve Validation Loss ({key})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Karışıklık Matrisi
    cm = confusion_matrix(y_true, y_val)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{opt} - Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(4)
    plt.xticks(tick_marks, label_map.keys(), rotation=45)
    plt.yticks(tick_marks, label_map.keys())
    for i in range(4):
        for j in range(4):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    plt.tight_layout()
    plt.show()