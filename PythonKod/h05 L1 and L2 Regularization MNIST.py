import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l1, l2

epoch = 5
batch_size = 32

# MNIST veri setini yükleniyor
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Görüntüler 0-1 arasına normalize ediliyor.
x_train, x_test = (x_train / 255.0), (x_test / 255.0)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Model oluşturuluyor
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
# L1 düzenlileştirme yapılıyor
model.add(Dense(128, activation='relu', kernel_regularizer=l1(0.001)))  
# L2 düzenlileştirme yapılıyor
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dense(10, activation='softmax'))

# Model derleniyor
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Model eğitiliyor
model.fit(x_train, y_train, 
          epochs=epoch, 
          batch_size=batch_size, 
          validation_split=0.2)

# Model değerlendiriliyor
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Kayıp: {loss}, Test Doğruluk: {accuracy}")
