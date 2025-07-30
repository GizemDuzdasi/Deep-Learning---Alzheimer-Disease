import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD, Nadam, RMSprop, Adadelta, Adagrad
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix
# Veri Yolu ve Boyutlar
dataset_path = r"data/50Augmented20SplitDataset"
img_size = 224

# Etiketler
label_map = {"NonDemented": 0, "VeryMildDemented": 1, "MildDemented": 2, "ModerateDemented": 3}

def load_data(folder_path):
    data, labels = [], []
    for category in os.listdir(folder_path):
        category_path = os.path.join(folder_path, category)
        if os.path.isdir(category_path):
            for file in os.listdir(category_path):
                file_path = os.path.join(category_path, file)
                try:
                    img = Image.open(file_path).convert('RGB')
                    img = img.resize((img_size, img_size))
                    data.append(np.array(img))
                    labels.append(label_map[category])
                except Exception as e:
                    print(f"Hata oluştu: {file_path}, {e}")
    return np.array(data) / 255.0, to_categorical(labels, num_classes=4)

# Eğitim ve Test Verisi
train_path = os.path.join(dataset_path, "Train")
test_path = os.path.join(dataset_path, "Test")

X_train, y_train = load_data(train_path)
X_test, y_test = load_data(test_path)
# DenseNet121 Modeli
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

# Model Fonksiyonu
def build_model(optimizer, dropout_rate=0.5):
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),       # Tabloya uygun
        BatchNormalization(),           # Evet (pooling katmanından sonra)
        Dense(128, activation='relu'),  # Dense katmanda 128 nöron, aktivasyon ReLU
        Dropout(dropout_rate),          # Dropout 0.5
        Dense(4, activation='softmax')  # Çıkış katmanı: Softmax
    ])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Optimizasyon Yöntemleri ve Öğrenme Oranları
optimizers = {
    'Adam': Adam(learning_rate=0.001),
    'SGD': SGD(learning_rate=0.01, momentum=0.9),
    'Nadam': Nadam(learning_rate=0.002),
    'RMSprop': RMSprop(learning_rate=0.001),
    'Adadelta': Adadelta(learning_rate=1.0),
    'Adagrad': Adagrad(learning_rate=0.01)
}
# Model özetini göster
base_model.summary()
results = {}
for optimizer_name, optimizer in optimizers.items():
    print(f"\n🔁 {optimizer_name} ile eğitim başlıyor...\n")
    model = build_model(optimizer)
    from keras.utils import plot_model

    plot_model(model, to_file='densenet121_plot.png', show_shapes=True, show_layer_names=True)

    history = model.fit(X_train, y_train, batch_size=32, epochs=20,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping, reduce_lr], verbose=1)

    y_pred = model.predict(X_test)
    y_val, y_true = np.argmax(y_pred, axis=1), np.argmax(y_test, axis=1)

    test_acc = accuracy_score(y_true, y_val)
    f1 = f1_score(y_true, y_val, average='weighted')
    recall = recall_score(y_true, y_val, average='weighted')

    valid_label_names = [k for k, v in label_map.items() if v in sorted(set(y_true))]

    report = classification_report(y_true, y_val, labels=[0, 1, 2, 3], target_names=valid_label_names, output_dict=True)

    results[optimizer_name] = {
        'test_accuracy': test_acc,
        'f1_score': f1,
        'recall_score': recall,
        'classification_report': report
    }

    print(f"✅ Test Accuracy: {test_acc:.4f}")
    print(f"✅ F1 Score: {f1:.4f}")
    print(f"✅ Recall: {recall:.4f}")
    print(f"✅ Classification Report:\n{classification_report(y_true, y_val, target_names=label_map.keys(), digits=4)}")

    cm = confusion_matrix(y_true, y_val)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=valid_label_names, yticklabels=valid_label_names)
    plt.xlabel('Tahmin')
    plt.ylabel('Gerçek')
    plt.title(f'{optimizer_name} - Confusion Matrix')
    plt.show()

    # Loss ve Accuracy Grafikleri
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{optimizer_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{optimizer_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
	# Sonuçların Özeti
for optimizer_name, result in results.items():
    print(f"\n📌 {optimizer_name} - Final Results")
    print(f"Test Accuracy: {result['test_accuracy']:.4f}")
    print(f"F1 Score: {result['f1_score']:.4f}")
    print(f"Recall Score: {result['recall_score']:.4f}")