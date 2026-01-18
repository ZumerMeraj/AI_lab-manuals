# next_word_5words.py

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical

# -----------------------------
# Step 1: Sample text dataset
# Replace this with your own text if needed
# -----------------------------
text = """Machine learning is a method of data analysis that automates analytical model building.
It is a branch of artificial intelligence based on the idea that systems can learn from data,
identify patterns and make decisions with minimal human intervention."""

# -----------------------------
# Step 2: Tokenize and create sequences
# -----------------------------
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

input_sequences = []
for line in text.split('.'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i+1])

max_len = max([len(seq) for seq in input_sequences])

# Pad sequences
input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

# Split X and y
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# One-hot encode y
y = to_categorical(y, num_classes=len(tokenizer.word_index)+1)

# -----------------------------
# Step 3: Build LSTM model
# -----------------------------
vocab_size = len(tokenizer.word_index) + 1

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=max_len-1))
model.add(LSTM(100))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# -----------------------------
# Step 4: Train the model
# -----------------------------
print("Training LSTM model...")
model.fit(X, y, epochs=200, verbose=1)  # Reduce epochs if training is slow

# -----------------------------
# Step 5: Function to generate 5 words sequentially
# -----------------------------
def generate_next_words(model, tokenizer, seed_text, max_len, num_words=5):
    text = seed_text
    for _ in range(num_words):
        # Convert current text to sequence
        token_list = tokenizer.texts_to_sequences([text])[0]
        # Pad to match model input
        token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
        # Predict next word
        predicted_index = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        # Map index back to word
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                next_word = word
                break
        # Append predicted word
        text += " " + next_word
    return text

# -----------------------------
# Step 6: User input
# -----------------------------
while True:
    seed_text = input("\nEnter a sentence (or type 'exit' to quit):\n> ")
    if seed_text.lower() == "exit":
        break
    generated_text = generate_next_words(model, tokenizer, seed_text, max_len, num_words=5)
    print("\nGenerated text:", generated_text)
