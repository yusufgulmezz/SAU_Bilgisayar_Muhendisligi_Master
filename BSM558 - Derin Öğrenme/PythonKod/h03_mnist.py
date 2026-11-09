import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models, layers, metrics, utils
from sklearn.model_selection import train_test_split

epoch = 20

# Mnist veri seti yükleniyor. Eğitim veriseti 60000 test veriseti 10000 görüntüdür.
(train_data, train_label), (test_data, test_label) = mnist.load_data()

train_data = train_data.reshape((60000,28*28)).astype('float32')/255
test_data = test_data.reshape((10000,28*28)).astype('float32')/255

train_label = utils.to_categorical(train_label)
test_label = utils.to_categorical(test_label)

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(28*28,)))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy','mae'])



# Eğitim ve doğrulama verileri ayrılıyor
x_train_partial, x_val, y_train_partial, y_val = train_test_split(train_data, 
                                                                  train_label, 
                                                                  test_size=0.25, 
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
    verbose=1)  

results = model.fit(x_train_partial,
                    y_train_partial,
                    epochs=epoch,
                    batch_size=256,
                    validation_data=(x_val, y_val))
                    #callbacks=[early_stopping, checkpoint])


# Eğitim geçmişi ekrana yazdırılıyor
print('\n\n')
formatted_data = "\n".join([f"{key}: {value}" for key, value in results.history.items()])
print(formatted_data)

# Eğitim ve doğrulama kayıplarının grafiği çiziliyor
epochs = range(1, len(results.history['loss']) + 1)
plt.plot(epochs, results.history['loss'], 'bo', label='Training loss')
plt.plot(epochs, results.history['val_loss'], 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, results.history['accuracy'], 'bo', label='Training Accuracy')
plt.plot(epochs, results.history['val_accuracy'], 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Test verisi kullanılarak modelin başarımı ölçülüyor
test_loss, test_acc, test_mae = model.evaluate(test_data, test_label)

# Test verisindeki kayıp ve doğruluk ekrana yazdırılıyor
print("\ntest_loss:{}\ntest_acc:{}\ntest_mae:{}\n".format(test_loss, test_acc, test_mae))

# Tek bir görüntüyü test etmek ve sonucu hem rakam olarak hem görüntü ile göstermek
tahmin = model.predict(test_data[0].reshape(1,28*28))
tahmin_sayi = np.argmax(tahmin)
test_image = test_data[0].reshape(28,28)
plt.imshow(test_image, cmap='gray')
plt.title(f"Label: {tahmin_sayi}")
plt.axis('off')  # Eksenleri kapatma
plt.show()

