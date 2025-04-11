import keras
import numpy as np
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt


Dense = keras.layers.Dense
Flatten = keras.layers.Flatten
Reshape = keras.layers.Reshape
Lambda = keras.layers.Lambda
Input = keras.layers.Input
Dropout = keras.layers.Dropout
BatchNormalization = keras.layers.BatchNormalization
Model = keras.models.Model

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255

x_test = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

hidden_dim = 2
batch_size = 60


def dropout_and_batch(x):
    return Dropout(0.3)(BatchNormalization()(x))


input_encoder = Input(shape=(28, 28, 1))
x = Flatten()(input_encoder)
x = Dense(256, activation="relu")(x)
x = dropout_and_batch(x)
x = Dense(128, activation="relu")(x)
x = dropout_and_batch(x)

# вектор математическое ожидание
z_mean = Dense(hidden_dim)(x)
# логарифм дисперсии
z_log_var = Dense(hidden_dim)(x)


def noiser(args):
    global z_mean, z_log_var
    z_mean, z_log_var = args
    N = tf.keras.backend.random_normal(
        shape=(batch_size, hidden_dim), mean=0.0, stddev=1.0
    )
    return tf.exp(z_log_var / 2) * N + z_mean


h = Lambda(noiser, output_shape=(hidden_dim,))([z_mean, z_log_var])

input_decoder = Input(shape=(hidden_dim,))
d = Dense(256, activation="relu")(input_decoder)
d = dropout_and_batch(d)
d = Dense(128, activation="relu")(d)
d = dropout_and_batch(d)
d = Dense(28 * 28, activation="sigmoid")(d)
decoder_output = Reshape((28, 28, 1))(d)

encoder = Model(input_encoder, h, name="encocer")
decoder = Model(input_decoder, decoder_output, name="decoder")

vae = Model(input_encoder, decoder(encoder(input_encoder)), name="vae")


def vae_loss(x, y):
    x = tf.reshape(x, shape=(batch_size, 28 * 28))
    y = tf.reshape(y, shape=(batch_size, 28 * 28))
    loss = tf.reduce_sum(tf.square(x - y), axis=-1)
    kl_loss = -0.5 * tf.reduce_sum(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1
    )
    return loss + kl_loss


print(vae.summary())

vae.compile(optimizer="adam", loss=vae_loss)
vae.fit(x_train, x_train, epochs=5, batch_size=batch_size, shuffle=True)

h = encoder.predict(x_test[:6000], batch_size=batch_size)
