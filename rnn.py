import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

max_len = 200
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

model = Sequential()

model.add(Embedding(input_dim=10000, output_dim=128, input_length=max_len))

model.add(SimpleRNN(64, activation='tanh'))

model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

loss, accuracy = model.evaluate(x_test, y_test)
print("Test accuracy:", accuracy)