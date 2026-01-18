import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
vocab_size = 10000
max_length = 200

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
X_train = pad_sequences(X_train, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)

print("Padded shape:", X_train.shape)
model = Sequential()

model.add(Embedding(vocab_size, 64, input_length=max_length))
model.add(SimpleRNN(64))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()
model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.2
)
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", round(accuracy * 100, 2), "%")
# Word index dictionary
word_index = imdb.get_word_index()

# Convert text to sequence
def encode_text(text):
    words = text.lower().split()
    encoded = []

    for word in words:
        if word in word_index and word_index[word] < vocab_size:
            encoded.append(word_index[word])
        else:
            encoded.append(2)  # unknown word

    padded = pad_sequences([encoded], maxlen=max_length)
    return padded


# Custom input
review = "this movie was amazing and very interesting"

encoded_review = encode_text(review)

prediction = model.predict(encoded_review)

if prediction[0][0] > 0.5:
    print("Review:", review)
    print("Sentiment: POSITIVE 😊")
else:
    print("Review:", review)
    print("Sentiment: NEGATIVE 😞")
