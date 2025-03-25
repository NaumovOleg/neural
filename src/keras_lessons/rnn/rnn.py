import numpy as np
import re
import keras
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

Sequential = keras.models.Sequential
Input = keras.layers.Input
Dense = keras.layers.Dense
SimpleRNN = keras.layers.SimpleRNN
Dropout = keras.layers.Dropout


with open("data_sets/text.txt", "r", encoding="utf-8") as file:
    text = file.read()
    text = text.replace("\ufeff", "")
    text = re.sub(r"[^А-я]", "", text)

num_characters = 34

tokenizer = Tokenizer(num_words=num_characters, char_level=True)
tokenizer.fit_on_texts(text.split())

data = tokenizer.texts_to_matrix(text)
inp_chars = 6
n = data.shape[0] - inp_chars

print(data.shape)

X = np.array([data[i : i + inp_chars, :] for i in range(n)])
y = data[inp_chars:]

model = Sequential(
    [
        Input(shape=(inp_chars, num_characters)),
        SimpleRNN(256, activation="tanh"),
        Dense(num_characters, activation="softmax"),
    ]
)

model.compile(loss="categorical_crossentropy", optimizer="adam")
print(model.summary())
history = model.fit(X, y, epochs=100)


def build_phrase(inp_string, str_len=50):
    for i in range(str_len):
        x = []
        for j in range(i, i + inp_chars):
            x.append(tokenizer.texts_to_matrix(inp_string[j]))
        x = np.array(x)
        inp = x.reshape((1, inp_chars, num_characters))
        predicted = model.predict(inp)
        d = tokenizer.index_word[predicted.argmax(axis=1)[0]]
        inp_string += d
    return inp_string


res = build_phrase("утренн")
print(res)
