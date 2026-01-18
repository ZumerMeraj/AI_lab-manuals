import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
# Sample text dataset (replace with your own)
text = """Machine learning is a method of data analysis that automates analytical model building.
It is a branch of artificial intelligence based on the idea that systems can learn from data,
identify patterns and make decisions with minimal human intervention."""

# Initialize tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

# Create input sequences
input_sequences = []
for line in text.split('.'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i+1])

# Find max sequence length
max_len = max([len(seq) for seq in input_sequences])

# Pad sequences
input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

# Split into X and y
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# One-hot encode y
from tensorflow.keras.utils import to_categorical
y = to_categorical(y, num_classes=len(tokenizer.word_index)+1)
vocab_size = len(tokenizer.word_index) + 1

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=max_len-1))
model.add(LSTM(100))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X, y, epochs=200, verbose=1)  # You can reduce epochs if it takes too long
def predict_next_word(model, tokenizer, seed_text, max_len):
    """
    Predict the next word for a given seed_text
    """
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
    predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
    
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            return word

# Example usage
seed_text = "machine learning is"
next_word = predict_next_word(model, tokenizer, seed_text, max_len)
print("Next word prediction:", next_word)
seed_text = "machine learning is"
for _ in range(5):  # predict 5 words
    next_word = predict_next_word(model, tokenizer, seed_text, max_len)
    seed_text += " " + next_word

print("Generated text:", seed_text)
