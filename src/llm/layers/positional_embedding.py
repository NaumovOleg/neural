import keras
import tensorflow as tf


Layer = keras.layers.Layer
Embedding = keras.layers.Embedding


class PositionalEmbedding(Layer):
    """Layer that adds positional embeddings to token embeddings."""

    def __init__(self, max_len, vocab_size, embed_dim):
        super().__init__()
        self.token_embed = Embedding(vocab_size, embed_dim)
        self.pos_embed = Embedding(max_len, embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        embedded = self.token_embed(x)
        pos = self.pos_embed(positions)
        return embedded + pos
