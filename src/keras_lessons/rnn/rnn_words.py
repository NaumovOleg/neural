import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

Sequential = keras.models.Sequential
Input = keras.layers.Input
Dense = keras.layers.Dense
SimpleRNN = keras.layers.SimpleRNN
Embedding = keras.layers.Embedding


with open("data_sets/text.txt", "r", encoding="utf-8") as file:
    text = file.read()
    text = text.replace("\ufeff", "")

max_words = 1000
tokenizer = Tokenizer(
    num_words=max_words,
    filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»',
    lower=True,
    split=" ",
    char_level=False,
)
tokenizer.fit_on_texts(text.split())


dist = list(tokenizer.word_counts.items())

data = tokenizer.texts_to_sequences([text])
sequences = np.array(data[0])


input_words_count = 3
n = len(sequences) - input_words_count


X = np.array([sequences[i : i + input_words_count] for i in range(n)])
Y = to_categorical(sequences[input_words_count:], num_classes=max_words)

model = Sequential(
    [
        Embedding(input_dim=max_words, output_dim=256),
        SimpleRNN(2000, activation="tanh"),
        Dense(max_words, activation="softmax"),
    ]
)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(X, Y, epochs=50)


def build_phrase(texts, str_len=20):
    res = texts
    data = tokenizer.texts_to_sequences([texts])[0]
    for i in range(str_len):
        # x = to_categorical(data[i: i + inp_words], num_classes=maxWordsCount)  # преобразуем в One-Hot-encoding
        # inp = x.reshape(1, inp_words, maxWordsCount)
        x = data[i : i + input_words_count]
        inp = np.expand_dims(x, axis=0)
        pred = model.predict(inp)
        indx = pred.argmax(axis=1)[0]
        data.append(indx)
        res += " " + tokenizer.index_word[indx]  # дописываем строку

    return res


res = build_phrase("позитив добавляет годы", 10)
print(res)
# test_data = np.array(tokenizer.texts_to_sequences(["позитив добавляет годы"]))
# predicted = model.predict(test_data)
# index = predicted.argmax(axis=1)[0]
# print(tokenizer.index_word[index])
