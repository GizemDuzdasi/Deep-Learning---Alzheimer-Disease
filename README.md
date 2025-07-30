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
- **Özellikler**:
  - Farklı nöron konfigürasyonları test edildi
  - 6 farklı optimizasyon algoritması karşılaştırıldı
  - Early stopping ve learning rate reduction kullanıldı

### 2. **DenseNet121**
- **Dosya**: `densenet121.py`
- **Veri Seti**: 50% Augmented 20% Split Dataset
- **Özellikler**:
  - Transfer öğrenme ile ImageNet ağırlıkları kullanıldı
  - Global Average Pooling ve Batch Normalization
  - 6 farklı optimizasyon algoritması test edildi

### 3. **EfficientNetB0**
- **Dosya**: `efficientnetb0.py`
- **Veri Seti**: Orijinal dataset (80-20 split)
- **Özellikler**:
  - Son 30 katman fine-tuning
  - Cosine Annealing Learning Rate Scheduler
  - Sınıf ağırlıkları ile dengesiz veri seti yönetimi

### 4. **MobileNetV2**
- **Dosya**: `mobilenetv2.py`
- **Veri Seti**: 50% Augmented 20% Split Dataset
- **Özellikler**:
  - Hafif ve mobil uyumlu mimari
  - Transfer öğrenme ile optimize edilmiş performans
  - 6 farklı optimizasyon algoritması karşılaştırması

### 5. **ResNet50**
- **Dosya**: `resnet50.py`
- **Veri Seti**: Orijinal dataset (80-20 split)
- **Özellikler**:
  - Son 20 katman fine-tuning
  - Cosine Annealing Learning Rate Scheduler
  - ModerateDemented sınıfı için özel ağırlık artırımı

### 6. **VGG16**
- **Dosya**: `vgg16.py`
- **Veri Seti**: Dataset 220 Split Dataset
- **Özellikler**:
  - Klasik VGG mimarisi
  - Transfer öğrenme ile ImageNet ağırlıkları
  - 6 farklı optimizasyon algoritması test edildi

### 7. **VGG19**
- **Dosya**: `vgg19.py`
- **Veri Seti**: Same Augmented 20% Split Dataset
- **Özellikler**:
  - VGG16'ya göre daha derin mimari
  - Transfer öğrenme ile optimize edilmiş performans
  - 6 farklı optimizasyon algoritması karşılaştırması

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
- Adam
- SGD (Stochastic Gradient Descent)
- Nadam
- RMSprop
- Adadelta
- Adagrad

### Değerlendirme Metrikleri:
- Accuracy (Doğruluk)
- F1 Score (Ağırlıklı ortalama)
- Recall Score (Ağırlıklı ortalama)
- Confusion Matrix
- Classification Report

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

## 🔍 Önemli Bulgular

- **Transfer Öğrenme**: ImageNet ağırlıkları ile eğitilen modeller daha iyi performans gösterdi
- **Optimizasyon**: Adam ve Nadam optimizasyonları genellikle en iyi sonuçları verdi
- **Veri Artırma**: Augmented veri setleri model performansını artırdı
- **Sınıf Dengesizliği**: ModerateDemented sınıfı için özel ağırlık artırımı gerekli

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