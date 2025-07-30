import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix
from tensorflow.keras.optimizers import Adam, SGD, Nadam, RMSprop, Adadelta, Adagrad

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

# 🧠 Transfer öğrenme modeli
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
for layer in base_model.layers:
    layer.trainable = False

# 🏗️ Model kurucu fonksiyon
def build_model(optimizer, dropout_rate=0.5):
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dropout(dropout_rate),
        Dense(4, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ⏱️ Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# 🧪 Optimizer seçenekleri (learning rate düşürüldü)
optimizers = {
    'Adam': Adam(learning_rate=0.0005),
    'SGD': SGD(learning_rate=0.005, momentum=0.9),
    'Nadam': Nadam(learning_rate=0.001),
    'RMSprop': RMSprop(learning_rate=0.0005),
    'Adadelta': Adadelta(learning_rate=0.5),
    'Adagrad': Adagrad(learning_rate=0.005)
}

# 🔍 Base model özeti
base_model.summary()

# 🔢 Eğitim ve değerlendirme
results = {}

for optimizer_name, optimizer in optimizers.items():
    print(f"\n🔄 {optimizer_name} Optimizer ile Model Eğitimi Başlıyor...\n")
    model = build_model(optimizer, dropout_rate=0.5)
    from keras.utils import plot_model

    plot_model(model, to_file='mobilenet_plot.png', show_shapes=True, show_layer_names=True)
    history = model.fit(X_train, y_train, batch_size=32, epochs=20,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping, reduce_lr], verbose=1)

    # 🔮 Tahmin
    y_pred = model.predict(X_test)
    y_val = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # 🎯 Metrikler
    test_acc = accuracy_score(y_true, y_val)
    f1 = f1_score(y_true, y_val, average='weighted')
    recall = recall_score(y_true, y_val, average='weighted')

    valid_label_names = [k for k, v in label_map.items() if v in set(y_true)]

    report = classification_report(y_true, y_val, labels=[0, 1, 2, 3],
                                   target_names=valid_label_names, output_dict=True, digits=4)

    results[optimizer_name] = {
        'test_accuracy': test_acc,
        'f1_score': f1,
        'recall_score': recall,
        'classification_report': report
    }

    # 🖨️ Rapor
    print(f"✅ Test Accuracy: {test_acc:.4f}")
    print(f"✅ F1 Score (Weighted Avg): {f1:.4f}")
    print(f"✅ Recall Score (Weighted Avg): {recall:.4f}")
    print(f"✅ Classification Report:\n{classification_report(y_true, y_val, target_names=label_map.keys(), digits=4)}")

    # 📊 Confusion Matrix
    cm = confusion_matrix(y_true, y_val)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=valid_label_names, yticklabels=valid_label_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{optimizer_name} - Confusion Matrix')
    plt.show()

    # 📈 Loss & Accuracy Grafikleri
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{optimizer_name} - Loss Grafiği')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{optimizer_name} - Accuracy Grafiği')
    plt.legend()
    plt.show()
    
    