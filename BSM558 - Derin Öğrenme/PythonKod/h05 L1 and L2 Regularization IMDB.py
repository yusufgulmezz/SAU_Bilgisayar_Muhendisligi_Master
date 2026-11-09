import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.regularizers import l1, l2

# IMDb veri seti yükleniyor
max_features = 100  # Kullanılacak en sık geçen kelime sayısı
max_len = 50  # Maksimum inceleme uzunluğu

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# pad_sequences ile 
# Kısa Dizileri Doldurma (Padding), 
# Uzun Dizileri Kesme (Truncating) işlemi yapılıyor.
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# Model oluşturuluyor
model = Sequential()
model.add(Embedding(max_features, 64, input_length=max_len))
# L1 düzenlileştirme ekleniyor
model.add(LSTM(64, kernel_regularizer=l1(0.001)))
# L2 düzenlileştirme ekleniyor
model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001)))  

# Model derleniyor
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Model eğitiliyor
model.fit(x_train, y_train, epochs=2, batch_size=32, validation_split=0.2)

# Model değerlendiriliyor
loss, accuracy = model.evaluate(x_test, y_test)

print(f"Test Kayıp: {loss}, Test Doğruluk: {accuracy}")
