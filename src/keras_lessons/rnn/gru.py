import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

GRU = keras.layers.GRU
Dense = keras.layers.Dense
Embedding = keras.layers.Embedding
Sequential = keras.models.Sequential
Tokenizer = tf.keras.preprocessing.text.Tokenizer
Adam = keras.optimizers.Adam

rng = np.random.default_rng(42)

with open("data_sets/text-good.txt", "r", encoding="utf-8") as f:
    text_true = f.readlines()
    text_true[0] = text_true[0].replace("\ufeff", "")

with open("data_sets/text-bad.txt", "r", encoding="utf-8") as f:
    text_false = f.readlines()
    text_false[0] = text_false[0].replace("\ufeff", "")


max_unique_words = 1500
max_text_len = 10
true_len = len(text_true)
false_len = len(text_false)
total_len = true_len + false_len

text = text_true + text_false

print(text[0:3])

tokenizer = Tokenizer(
    num_words=max_unique_words,
    filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»',
    lower=True,
    split=" ",
    char_level=False,
)
tokenizer.fit_on_texts(text)
dist = list(tokenizer.word_counts.items())
sequences = tokenizer.texts_to_sequences(text)
print("++++++++++", text[0], sequences[0])
pad_sequences_list = pad_sequences(sequences, maxlen=max_text_len)

X = pad_sequences_list
Y = np.array([[1, 0]] * true_len + [[0, 1]] * false_len)

indeces = rng.choice(X.shape[0], size=X.shape[0], replace=False)
X = X[indeces]
Y = Y[indeces]

model = Sequential(
    [
        Embedding(
            max_unique_words,
            max_text_len,
            input_length=max_text_len,
        ),
        GRU(128, return_sequences=True),
        GRU(64),
        Dense(2, activation="softmax"),
    ]
)
adam_optimizer = Adam(learning_rate=0.0001)
model.compile(
    optimizer=adam_optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)
history = model.fit(X, Y, epochs=50)
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))


# def sequence_to_text(list_of_indices):
#     words = [reverse_word_map.get(letter) for letter in list_of_indices]
#     return words


t = np.array(["чудеса рождаются из убеждений"])
print(t, end="\n")
data = tokenizer.texts_to_sequences(t)
print(data)
data_pad = pad_sequences(data, maxlen=max_text_len)
print(data_pad)

res = model.predict(data_pad)
print(res, sep="\n")
