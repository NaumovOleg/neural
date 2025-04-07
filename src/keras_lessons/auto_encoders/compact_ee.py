import keras
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

Input = keras.layers.Input
Dense = keras.layers.Dense
Reshape = keras.layers.Reshape
Flatten = keras.layers.Flatten
Model = keras.models.Model

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))

encoder_input = Input(shape=(28, 28, 1))
x = Flatten()(encoder_input)
x = Dense(128, activation="relu")(x)
x = Dense(64, activation="relu")(x)
encoder_layer = Dense(4, activation="linear")(x)

input_decoder = Input(shape=(4,))
d = Dense(64, activation="relu")(input_decoder)
d = Dense(28 * 28, activation="sigmoid")(d)
decoder_layer = Reshape((28, 28, 1))(d)

encoder = Model(encoder_input, encoder_layer)
decoder = Model(input_decoder, decoder_layer)
auto_encoder = Model(encoder_input, decoder(encoder(encoder_input)))

auto_encoder.compile(optimizer="adam", loss="mse")
auto_encoder.fit(x_train, x_train, epochs=10, batch_size=64)

h = encoder.predict(x_test)
print(x_train[0].shape, h.shape)


fig, axes = plt.subplots(2, 2, figsize=(15, 6))


def show_plot():
    axes[0, 0].scatter(h[:, 0], h[:, 1])
    # axes[1, 0].imshow(h.squeeze(), cmap="gray")
    img = decoder.predict(np.expand_dims(h[0], axis=0))
    origin_image = np.expand_dims(x_test[0], axis=0)
    axes[1, 0].imshow(img.squeeze(), cmap="gray")
    axes[1, 1].imshow(origin_image.squeeze(), cmap="gray")


# axes[1, 0].axis("off")


show_plot()
plt.show()
