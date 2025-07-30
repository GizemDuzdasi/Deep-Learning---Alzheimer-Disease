# 🧠 Alzheimer Hastalığı Teşhisi - Derin Öğrenme Projesi

Bu proje, Alzheimer hastalığının erken teşhisi için çeşitli derin öğrenme modellerini kullanarak beyin MRI görüntülerini analiz eden kapsamlı bir çalışmadır.

## 📋 Proje Özeti

Alzheimer hastalığı, demansın en yaygın türüdür ve erken teşhis büyük önem taşır. Bu proje, beyin MRI görüntülerini analiz ederek Alzheimer hastalığının farklı aşamalarını (NonDemented, VeryMildDemented, MildDemented, ModerateDemented) sınıflandırmak için çeşitli derin öğrenme modellerini test eder.

## 🎯 Hedefler

- Alzheimer hastalığının 4 farklı aşamasını sınıflandırma
- Farklı derin öğrenme mimarilerinin performansını karşılaştırma
- Transfer öğrenme tekniklerini kullanarak model performansını optimize etme
- Çeşitli optimizasyon algoritmalarının etkisini analiz etme

## 🏗️ Kullanılan Modeller

### 1. **CNN (Convolutional Neural Network)**
- **Dosya**: `cnn.py`
- **Veri Seti**: Orijinal dataset (70-30 split)
- **Görüntü Formatı**: Grayscale (1 kanal)
- **Özellikler**:
  - Farklı nöron konfigürasyonları test edildi: (32,32), (64,64), (32,32,32,32), (32,32,64,64), (64,64,64,64)
  - 6 farklı optimizasyon algoritması karşılaştırıldı: Adam, RMSprop, SGD, Adagrad, Adadelta, Nadam
  - Early stopping ve learning rate reduction kullanıldı
  - Dropout katmanları ile overfitting önlendi

### 2. **DenseNet121**
- **Dosya**: `densenet121.py`
- **Veri Seti**: 50% Augmented 20% Split Dataset
- **Görüntü Formatı**: RGB (3 kanal)
- **Özellikler**:
  - Transfer öğrenme ile ImageNet ağırlıkları kullanıldı
  - Global Average Pooling ve Batch Normalization
  - 6 farklı optimizasyon algoritması test edildi
  - Dropout rate: 0.5

### 3. **EfficientNetB0**
- **Dosya**: `efficientnetb0.py`
- **Veri Seti**: Orijinal dataset (80-20 split)
- **Görüntü Formatı**: RGB (3 kanal)
- **Özellikler**:
  - Son 30 katman fine-tuning
  - Cosine Annealing Learning Rate Scheduler
  - Sınıf ağırlıkları ile dengesiz veri seti yönetimi
  - ModerateDemented sınıfı için 10x ağırlık artırımı
  - Dropout rate: 0.2

### 4. **MobileNetV2**
- **Dosya**: `mobilenetv2.py`
- **Veri Seti**: 50% Augmented 20% Split Dataset
- **Görüntü Formatı**: RGB (3 kanal)
- **Özellikler**:
  - Hafif ve mobil uyumlu mimari
  - Transfer öğrenme ile optimize edilmiş performans
  - 6 farklı optimizasyon algoritması karşılaştırması
  - Dropout rate: 0.5
  - Dense katmanında 256 nöron

### 5. **ResNet50**
- **Dosya**: `resnet50.py`
- **Veri Seti**: Orijinal dataset (80-20 split)
- **Görüntü Formatı**: RGB (3 kanal)
- **Özellikler**:
  - Son 20 katman fine-tuning
  - Cosine Annealing Learning Rate Scheduler
  - ModerateDemented sınıfı için 25x ağırlık artırımı
  - Dropout rate: 0.5
  - Batch size: 4 (düşük batch size)

### 6. **VGG16**
- **Dosya**: `vgg16.py`
- **Veri Seti**: Dataset 220 Split Dataset
- **Görüntü Formatı**: RGB (3 kanal)
- **Özellikler**:
  - Klasik VGG mimarisi
  - Transfer öğrenme ile ImageNet ağırlıkları
  - 6 farklı optimizasyon algoritması test edildi
  - Dropout rate: 0.3
  - Dense katmanında 128 nöron

### 7. **VGG19**
- **Dosya**: `vgg19.py`
- **Veri Seti**: Same Augmented 20% Split Dataset
- **Görüntü Formatı**: RGB (3 kanal)
- **Özellikler**:
  - VGG16'ya göre daha derin mimari
  - Transfer öğrenme ile optimize edilmiş performans
  - 6 farklı optimizasyon algoritması karşılaştırması
  - Dropout rate: 0.5
  - Dense katmanında 128 nöron

## 📊 Veri Seti

Proje, Alzheimer hastalığının 4 farklı aşamasını içeren beyin MRI görüntülerini kullanır:

- **NonDemented**: Demans olmayan
- **VeryMildDemented**: Çok hafif demans
- **MildDemented**: Hafif demans
- **ModerateDemented**: Orta demans

### Veri Seti Özellikleri:
- **Görüntü Boyutu**: 224x224 piksel
- **Renk Kanalı**: RGB (3 kanal) ve Grayscale (1 kanal)
- **Veri Artırma**: Bazı modellerde data augmentation kullanıldı
- **Split Oranları**: 70-30, 80-20, 50% Augmented gibi farklı konfigürasyonlar

## 🔧 Teknik Özellikler

### Kullanılan Kütüphaneler:
- **TensorFlow/Keras**: Derin öğrenme modelleri
- **NumPy**: Sayısal işlemler
- **PIL**: Görüntü işleme
- **Matplotlib/Seaborn**: Görselleştirme
- **Scikit-learn**: Metrikler ve veri bölme
- **Google Colab**: Çalışma ortamı

### Optimizasyon Algoritmaları:
- **Adam**: Learning rate 0.001 (CNN, DenseNet, VGG16, VGG19), 0.0001 (EfficientNet, ResNet), 0.0005 (MobileNet)
- **SGD**: Learning rate 0.01 (CNN, DenseNet, VGG16, VGG19), 0.00005 (EfficientNet, ResNet), 0.005 (MobileNet)
- **Nadam**: Learning rate 0.002 (CNN, DenseNet, VGG16, VGG19), 0.0001 (EfficientNet, ResNet), 0.001 (MobileNet)
- **RMSprop**: Learning rate 0.001 (CNN, DenseNet, VGG16, VGG19), 0.0001 (EfficientNet, ResNet), 0.0005 (MobileNet)
- **Adadelta**: Learning rate 1.0 (CNN, DenseNet, VGG16, VGG19), 1.0 (EfficientNet, ResNet), 0.5 (MobileNet)
- **Adagrad**: Learning rate 0.01 (CNN, DenseNet, VGG16, VGG19), 0.001 (EfficientNet, ResNet), 0.005 (MobileNet)

### Değerlendirme Metrikleri:
- **Accuracy (Doğruluk)**: Test seti üzerinde doğru tahmin oranı
- **F1 Score (Ağırlıklı ortalama)**: Precision ve Recall'ın harmonik ortalaması
- **Recall Score (Ağırlıklı ortalama)**: Gerçek pozitiflerin doğru tahmin edilme oranı
- **Confusion Matrix**: Sınıf bazında karışıklık matrisi
- **Classification Report**: Detaylı sınıf bazında performans raporu

## 🚀 Kurulum ve Kullanım

### Gereksinimler:
```bash
pip install tensorflow
pip install numpy
pip install pillow
pip install matplotlib
pip install seaborn
pip install scikit-learn
```

### Çalıştırma:
Her model dosyasını ayrı ayrı çalıştırabilirsiniz:

```bash
python cnn.py
python densenet121.py
python efficientnetb0.py
python mobilenetv2.py
python resnet50.py
python vgg16.py
python vgg19.py
```

## 📈 Sonuçlar

Her model için aşağıdaki analizler yapılmıştır:

1. **Model Performansı**: Test accuracy, F1 score, recall değerleri
2. **Optimizasyon Karşılaştırması**: 6 farklı optimizasyon algoritmasının performansı
3. **Görselleştirme**: 
   - Confusion Matrix
   - Loss ve Accuracy grafikleri
   - Model mimarisi diyagramları
4. **Detaylı Raporlar**: Classification report ile sınıf bazında performans

## 📁 Proje Yapısı

```
Deep-Learning---Alzheimer-Disease/
├── cnn.py                 # CNN modeli (Grayscale, 70-30 split)
├── densenet121.py         # DenseNet121 modeli (RGB, Augmented dataset)
├── efficientnetb0.py      # EfficientNetB0 modeli (RGB, 80-20 split)
├── mobilenetv2.py         # MobileNetV2 modeli (RGB, Augmented dataset)
├── resnet50.py            # ResNet50 modeli (RGB, 80-20 split)
├── vgg16.py               # VGG16 modeli (RGB, 220 split dataset)
├── vgg19.py               # VGG19 modeli (RGB, Augmented dataset)
├── requirements.txt        # Gerekli Python kütüphaneleri
├── README.md              # Proje dokümantasyonu
└── bitirme.pdf            # Bitirme tezi
```

## 🔧 Model Konfigürasyonları

### CNN Modeli:
- **Giriş**: 224x224x1 (Grayscale)
- **Konvolüsyon Katmanları**: Farklı nöron sayıları test edildi
- **Dropout**: 0.2, 0.25, 0.3 oranlarında
- **Dense Katmanları**: 100 ve 50 nöron

### Transfer Learning Modelleri:
- **Base Model**: ImageNet ağırlıkları ile önceden eğitilmiş
- **Fine-tuning**: EfficientNet (son 30 katman), ResNet (son 20 katman)
- **Global Average Pooling**: Özellik çıkarımı için
- **Batch Normalization**: Eğitim stabilizasyonu için
- **Dropout**: Overfitting önleme için
- **Dense Katmanları**: 128-256 nöron arası

## 🔍 Önemli Bulgular

- **Transfer Öğrenme**: ImageNet ağırlıkları ile eğitilen modeller daha iyi performans gösterdi
- **Optimizasyon**: Adam ve Nadam optimizasyonları genellikle en iyi sonuçları verdi
- **Veri Artırma**: Augmented veri setleri model performansını artırdı
- **Sınıf Dengesizliği**: ModerateDemented sınıfı için özel ağırlık artırımı gerekli
- **Fine-tuning**: EfficientNet ve ResNet modellerinde son katmanların eğitilebilir yapılması performansı artırdı
- **Cosine Annealing**: SGD optimizasyonu ile birlikte kullanılan cosine annealing scheduler daha stabil eğitim sağladı
- **Batch Size**: ResNet50'de düşük batch size (4) kullanılması memory optimizasyonu sağladı
- **Dropout**: Farklı modellerde farklı dropout oranları (0.2-0.5) kullanılarak overfitting önlendi

## 📝 Lisans

Bu proje eğitim ve araştırma amaçlı geliştirilmiştir.

## 👨‍💻 Geliştirici

Bu proje, Alzheimer hastalığı teşhisi için derin öğrenme tekniklerini araştırmak ve karşılaştırmak amacıyla geliştirilmiştir.

## 🤝 Katkıda Bulunma

Projeye katkıda bulunmak için:
1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

## 📞 İletişim

Sorularınız için GitHub üzerinden issue açabilirsiniz.

---

**Not**: Bu proje eğitim amaçlıdır ve tıbbi teşhis için kullanılmamalıdır. Gerçek tıbbi uygulamalar için uzman doktorların değerlendirmesi gereklidir. 