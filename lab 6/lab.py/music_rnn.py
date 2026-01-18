import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.utils import to_categorical

# -----------------------------
# 1. Predefined small note sequence
# -----------------------------
notes = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"] * 10  # repeated for more data

# Mapping notes to integers
pitchnames = sorted(set(notes))
n_notes = len(pitchnames)
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

# -----------------------------
# 2. Prepare sequences
# -----------------------------
sequence_length = 5
network_input = []
network_output = []

for i in range(len(notes) - sequence_length):
    seq_in = notes[i:i + sequence_length]
    seq_out = notes[i + sequence_length]
    network_input.append([note_to_int[n] for n in seq_in])
    network_output.append(note_to_int[seq_out])

network_input = np.array(network_input)
network_output = to_categorical(network_output, num_classes=n_notes)

# -----------------------------
# 3. Build RNN model
# -----------------------------
model = Sequential()
model.add(SimpleRNN(64, input_shape=(sequence_length, 1)))
model.add(Dense(n_notes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Reshape input for RNN: (samples, timesteps, features)
network_input = network_input.reshape((network_input.shape[0], sequence_length, 1))

# -----------------------------
# 4. Train model
# -----------------------------
model.fit(network_input, network_output, epochs=50, batch_size=16, verbose=1)

# -----------------------------
# 5. Generate new note sequence
# -----------------------------
start = np.random.randint(0, len(network_input)-1)
pattern = network_input[start]
prediction_output = []

for note_index in range(20):  # generate 20 notes
    input_seq = pattern.reshape(1, sequence_length, 1)
    prediction = model.predict(input_seq, verbose=0)
    index = np.argmax(prediction)
    result = pitchnames[index]
    prediction_output.append(result)

    pattern = np.append(pattern[1:], [[index]], axis=0)

print("Generated Note Sequence:")
print(prediction_output)
