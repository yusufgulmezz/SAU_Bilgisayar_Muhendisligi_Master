# Gerekli kütüphaneler yükleniyor
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers, optimizers, losses, metrics
from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split

# Modelin kaç epoch boyunca eğitileceği belirtiliyor
epoch = 20

# IMDb veri seti yükleniyor, en sık geçen 10.000 kelimeye göre sınırlama yapılıyor
(train_data, train_label), (test_data, test_label) = imdb.load_data(num_words=10000)

# Veri setinin boyutları ekrana yazdırılıyor
print("x_train length: {}\ny_train length: {}\n"
      "x_test length: {}\ny_test length: {}".format(len(train_data), 
                                                    len(train_label), 
                                                    len(test_data), 
                                                    len(test_label)))

# Eğitim verisinin ve etiketi örneği ekrana yazdırılıyor
print("x_train ilk verisi:", train_data[1100], "\n")
print("y_train ilk verisi:", train_label[1100], "\n")

# IMDb veri setindeki kelimeleri sayısal indekslere dönüştüren sözlük alınıyor
word_index = imdb.get_word_index()

# Kelime indekslerini 3 artırarak özel karakterlerin ayrılmasını sağlıyoruz
word_index = {k: (v + 3) for k, v in word_index.items()}

# İndekslerden kelimelere geçiş yapmak için sözlük oluşturuluyor
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Eğitim verisinin ilk yorumunun kelime olarak geri çevrilmesi
str = [reverse_word_index.get(i, '?') for i in train_data[0]]
print(str)

# Yorumların vektörlere dönüştürülmesi için bir fonksiyon tanımlanıyor
def vectorize_sequences(sequences, dimension=10000):
    # Sıfırlardan oluşan 10,000 elemanlı bir matris oluşturuluyor
    results = np.zeros((len(sequences), dimension))
    # Her bir yorum için, kullanılan kelimelerin indekslerine 1 atanıyor
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# Eğitim ve test verileri vektöre dönüştürülüyor
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Etiket verileri (sınıflar) float32 tipine dönüştürülüyor
y_train = np.asarray(train_label).astype('float32')
y_test = np.asarray(test_label).astype('float32')

# Eğitim verisinin ilk örneği sıralanarak bir sütun vektörü olarak yazdırılıyor
y1 = np.sort(train_data[0]).reshape(len(train_data[0]), 1)
print(y1)

# Eğitim verisinin ilk örneği (vektöre dönüştürülmüş) ilk 50 elemanı yazdırılıyor
y2 = x_train[0].reshape(len(x_train[0]), 1)
print(y2[0:50])

# Modelin mimarisi tanımlanıyor
model = models.Sequential()
# İlk katman 16 nörondan oluşuyor ve 'relu' aktivasyon fonksiyonu kullanıyor
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
# İkinci katman da 16 nöron ve 'relu' aktivasyonu kullanıyor
model.add(layers.Dense(16, activation='relu'))
# Çıkış katmanı tek nörondan oluşuyor ve 'sigmoid' aktivasyon fonksiyonu kullanıyor
model.add(layers.Dense(1, activation='sigmoid'))

# Model derleniyor
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
              loss=losses.BinaryCrossentropy(),
              metrics=metrics.BinaryAccuracy())

# doğrulama veriseti kullanılmadan eğitim yapılacaksa
#history = model.fit(x_train, y_train, epochs=epoch, batch_size=500)


# Eğitim ve doğrulama verileri ayrılıyor
x_train_partial, x_val, y_train_partial, y_val = train_test_split(x_train, 
                                                                  y_train, 
                                                                  test_size=0.5, 
                                                                  random_state=42)

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', 
                               patience=3, 
                               restore_best_weights=True)

# ModelCheckpoint callback'i tanımlanıyor
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint(
    'model_epoch_{epoch:02d}.h5',  # Her epoch için ayrı dosya
    save_freq='epoch',            # Her epoch sonunda kaydedilir
    save_weights_only=False,      # Modelin tam hali kaydedilir (ağırlıklar ve yapı)
    monitor='val_loss',           # Takip edilecek metrik
    save_best_only=False,         # En iyi model değil, her epoch'ta kaydedilecek
    verbose=1                     # Her kaydetme işlemini bildirir
)
# Model eğitiliyor
history = model.fit(x_train_partial, 
                    y_train_partial, 
                    epochs=epoch,
                    batch_size=500,
                    validation_data=(x_val, y_val),
                    callbacks=[early_stopping, checkpoint])

# Eğitim geçmişi ekrana yazdırılıyor
print('\n\n')
formatted_data = "\n".join([f"{key}: {value}" for key, value in history.history.items()])
print(formatted_data)

# Eğitim ve doğrulama kayıplarının grafiği çiziliyor
epochs = range(1, len(history.history['loss']) + 1)
plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, history.history['binary_accuracy'], 'bo', label='Training Accuracy')
plt.plot(epochs, history.history['val_binary_accuracy'], 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Test verisi kullanılarak modelin başarımı ölçülüyor
test_loss, test_acc = model.evaluate(x_test, y_test)

# Test verisindeki kayıp ve doğruluk ekrana yazdırılıyor
print("\ntest_loss:{}\ntest_acc:{}".format(test_loss, test_acc))
