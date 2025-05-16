import tensorflow as tf
import keras
from corpus import corpus

Tokenizer = tf.keras.preprocessing.text.Tokenizer


tokenizer = Tokenizer(filters="", lower=True, oov_token="<unk>")
tokenizer.fit_on_texts(corpus)

vocab_size = len(tokenizer.word_index) + 1
print("Vocab size:", vocab_size)
