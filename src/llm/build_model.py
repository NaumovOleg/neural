import tensorflow as tf
from layers import TransformerDecoderBlock, PositionalEmbedding

import keras

Input = keras.layers.Input
GlobalAveragePooling1D = keras.layers.GlobalAveragePooling1D
Dense = keras.layers.Dense
Model = keras.Model


def build_model(maxlen, vocab_size, embed_dim=64, heads=2, ff_dim=128):
    inputs = Input(shape=(maxlen - 1,))
    x = PositionalEmbedding(maxlen, vocab_size, embed_dim)(inputs)
    x = TransformerDecoderBlock(embed_dim, heads, ff_dim)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(vocab_size, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model
