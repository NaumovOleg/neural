import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

GRU = keras.layers.GRU
Dense = keras.layers.Dense
Sequential = keras.models.Sequential
Tokenizer = tf.keras.preprocessing.text.Tokenizer


with open("data_sets/text-good.txt", "r", encoding="utf-8") as f:
    text_true = f.readlines()
    text_true[0] = text_true[0].replace("\ufeff", "")

with open("data_sets/text-bad.txt", "r", encoding="utf-8") as f:
    text_false = f.readlines()
    text_false[0] = text_false[0].replace("\ufeff", "")

print(text_false[0])

texts = text_true + text_false
count_true = len(text_true)
count_false = len(text_false)
total_lines = count_true + count_false

maxWordsCount = 1000
tokenizer = Tokenizer(
    num_words=maxWordsCount,
    filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»',
    lower=True,
    split=" ",
    char_level=False,
)
tokenizer.fit_on_texts(texts)

dist = list(tokenizer.word_counts.items())


max_text_len = 10
data = tokenizer.texts_to_sequences(texts)
data_pad = pad_sequences(data, maxlen=max_text_len)
print(data_pad[0:3])

# max_unique_words = 1000
# max_text_len = 10
# true_len = len(text_true)
# false_len = len(text_false)
# total_len = true_len + false_len

# text = text_true + text_false

# tokenizer = Tokenizer(
#     num_words=max_unique_words,
#     filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»',
#     lower=True,
#     split=" ",
#     char_level=False,
# )
# tokenizer.fit_on_texts(text.split())

# dist = list(tokenizer.word_counts.items())
# sequences = tokenizer.texts_to_sequences([text])
# # sequences = np.array(data[0])
# pad_sequences = pad_sequences(sequences, maxlen=max_text_len)
# print(pad_sequences[0:3])
