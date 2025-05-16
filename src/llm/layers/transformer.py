import keras
import tensorflow as tf

Layer = keras.layers.Layer
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
LayerNormalization = keras.layers.LayerNormalization
MultiHeadAttention = keras.layers.MultiHeadAttention
Sequential = keras.Sequential


class TransformerDecoderBlock(Layer):
    """
    Transformer decoder block consisting of two multi-head attention layers and a feed-forward layer.
    """

    def __init__(self, embed_dim, heads, ff_dim, dropout=0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=heads, key_dim=embed_dim)
        self.ffn = Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim)])
        self.ln1 = LayerNormalization()
        self.ln2 = LayerNormalization()
        self.dropout = Dropout(dropout)

    def call(self, x, training=False):
        # Causal mask
        mask = tf.linalg.band_part(tf.ones((tf.shape(x)[1], tf.shape(x)[1])), -1, 0)
        attention = self.att(x, x, attention_mask=mask)
        x = self.ln1(x + attention)
        ffn_out = self.ffn(x)
        return self.ln2(x + self.dropout(ffn_out, training=training))
