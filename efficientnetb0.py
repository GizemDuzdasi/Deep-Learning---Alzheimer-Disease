#orjinal dataset 80-20
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.optimizers import Adam, SGD, Nadam, RMSprop, Adadelta, Adagrad
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from google.colab import drive
from collections import Counter
from keras.utils import plot_model
import math

# Google Drive'ı bağla
drive.mount('/content/drive')

# Veri yolu
dataset_path = r"/content/drive/MyDrive/OrjinalDataset"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Veri yolu bulunamadı: {dataset_path}")
drive_path = '/content/drive/MyDrive/EfficientNetB0-20Split_plot.png'

# Sabitler
img_size = 224

# Veriyi yükleme fonksiyonu
def load_data(dataset_path):
    data, labels = [], []
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):
            for file in os.listdir(category_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(category_path, file)
                    try:
                        img = Image.open(file_path).convert('RGB')
                        img = img.resize((img_size, img_size))
                        data.append(np.array(img))
                        labels.append(category.strip())
                    except Exception as e:
                        print(f"Hata oluştu: {file_path}, {e}")
    return np.array(data), labels

# Veri yükle
X, y = load_data(dataset_path)
print("X shape before reshape:", X.shape)
if X.shape[1:] != (img_size, img_size, 3):
    X = X.reshape(X.shape[0], img_size, img_size, 3)
print("X shape after reshape:", X.shape)

# Etiket haritası
label_map = {"NonDemented": 0, "VeryMildDemented": 1, "MildDemented": 2, "ModerateDemented": 3}
y_num = [label_map.get(label, -1) for label in y]
if -1 in y_num:
    raise ValueError("Geçersiz etiketler bulundu!")
y = to_categorical(y_num, num_classes=4)

print("Son Sınıf Dağılımı:", Counter(y_num))

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y_num)

# Sınıf ağırlıkları
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_num), y=y_num)
class_weights = dict(enumerate(class_weights))
class_weights[3] *= 10.0  # ModerateDemented için ağırlığı artır
print("Ayarlanmış Sınıf Ağırlıkları:", class_weights)

# EfficientNetB0 modelini yükle
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

# Son 30 katmanı eğitilebilir yap
for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Model oluşturma
def build_model(optimizer, dropout_rate=0.2):
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(128, activation='relu'),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dropout(dropout_rate / 2),
        Dense(4, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1)

# Cosine Annealing Learning Rate Scheduler (SGD için)
def cosine_annealing(epoch, lr):
    epochs = 50
    initial_lr = 0.00005
    return initial_lr * (1 + math.cos(math.pi * epoch / epochs)) / 2

lr_scheduler = LearningRateScheduler(cosine_annealing)

# Optimizasyonlar
optimizers = {
    'SGD': SGD(learning_rate=0.00005, momentum=0.95),
    'Nadam': Nadam(learning_rate=0.0001),
    'Adam': Adam(learning_rate=0.0001),
    'RMSprop': RMSprop(learning_rate=0.0001),
    'Adagrad': Adagrad(learning_rate=0.001),
    'Adadelta': Adadelta(learning_rate=1.0),
}


results = {}
for optimizer_name, optimizer in optimizers.items():
    print(f"\n🔁 {optimizer_name} ile eğitim başlıyor...\n")
    model = build_model(optimizer, dropout_rate=0.2)

    callbacks = [early_stopping, reduce_lr]
    if optimizer_name == 'SGD':
        callbacks = [early_stopping, lr_scheduler]  # SGD için Cosine Annealing
    plot_model(model, to_file=drive_path, show_shapes=True, show_layer_names=True)

    history = model.fit(X_train, y_train, batch_size=16, epochs=30,
                        validation_data=(X_test, y_test),
                        callbacks=callbacks, verbose=1)

    y_pred = model.predict(X_test)
    y_val, y_true = np.argmax(y_pred, axis=1), np.argmax(y_test, axis=1)
    validation_acc = max(history.history['val_accuracy'])

    test_acc = accuracy_score(y_true, y_val)
    f1 = f1_score(y_true, y_val, average='weighted')
    recall = recall_score(y_true, y_val, average='weighted')

    valid_label_names = [k for k, v in label_map.items() if v in sorted(set(y_true))]

    report = classification_report(y_true, y_val, labels=[0, 1, 2, 3], target_names=valid_label_names, output_dict=True)

    results[optimizer_name] = {
        'val_accuracy': validation_acc,
        'test_accuracy': test_acc,
        'f1_score': f1,
        'recall_score': recall,
        'classification_report': report
    }

    print(f"✅ Best Validation Accuracy: {validation_acc:.4f}")
    print(f"✅ Test Accuracy: {test_acc:.4f}")
    print(f"✅ F1 Score: {f1:.4f}")
    print(f"✅ Recall: {recall:.4f}")
    print(f"✅ Classification Report:\n{classification_report(y_true, y_val, target_names=valid_label_names, digits=4)}")

    cm = confusion_matrix(y_true, y_val)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=valid_label_names, yticklabels=valid_label_names)
    plt.xlabel('Tahmin')
    plt.ylabel('Gerçek')
    plt.title(f'{optimizer_name} - Confusion Matrix')
    plt.show()

    # Loss ve Accuracy grafikleri
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{optimizer_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, None)  # y-eksenini 0-1 aralığına sabitle
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{optimizer_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(None, 1)  # y-eksenini 0-1 aralığına sabitle
    plt.legend()
    plt.show()

# Final Sonuçlar
for optimizer_name, result in results.items():
    print(f"\n📌 {optimizer_name} - Final Results")
    print(f"Best Validation Accuracy: {validation_acc:.4f}")
    print(f"Test Accuracy: {result['test_accuracy']:.4f}")
    print(f"F1 Score: {result['f1_score']:.4f}")
    print(f"Recall Score: {result['recall_score']:.4f}")