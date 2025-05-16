import keras
from tokenizer import tokenizer, vocab_size
from corpus import corpus
import tensorflow as tf


pad_sequences = keras.preprocessing.sequence.pad_sequences

sequences = []

for line in corpus:
    tokens = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(tokens)):
        seq = tokens[: i + 1]
        sequences.append(seq)

# Pad sequences
maxlen = max(len(x) for x in sequences)
sequences = pad_sequences(sequences, maxlen=maxlen, padding="pre")

X = sequences[:, :-1]
y = sequences[:, -1]

y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)
