import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# 1. Load small text dataset
# -----------------------------

text = """
the sun is shining
the sun is bright
the moon is shining
the stars are bright
the sun and moon shine
"""

# Clean text
text = text.lower()
# -----------------------------
# 2. Tokenize text
# -----------------------------

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1
print("Total words:", total_words)

# -----------------------------
# Create sequences (3 words -> next word)
# -----------------------------

input_sequences = []

for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(3, len(token_list)):
        seq = token_list[i-3:i+1]
        input_sequences.append(seq)

input_sequences = np.array(input_sequences)
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

y = tf.keras.utils.to_categorical(y, num_classes=total_words)

print("X shape:", X.shape)
print("y shape:", y.shape)
# -----------------------------
# 3. Build RNN Model
# -----------------------------

model = Sequential([
    Embedding(total_words, 10, input_length=3),
    SimpleRNN(64),
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# -----------------------------
# 4. Train model
# -----------------------------

model.fit(X, y, epochs=300, verbose=1)
# -----------------------------
# 5. Prediction function
# -----------------------------

def predict_next_word(text_input):
    token_list = tokenizer.texts_to_sequences([text_input])[0]
    token_list = pad_sequences([token_list], maxlen=3, padding='pre')

    prediction = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(prediction)

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word


# Test
test_input = "the sun is"
print("Input:", test_input)
print("Predicted Next Word:", predict_next_word(test_input))
