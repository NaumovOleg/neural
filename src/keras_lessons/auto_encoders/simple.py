import keras
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

Dense = keras.layers.Dense
Input = keras.layers.Input
Model = keras.models.Model
Flatten = keras.layers.Flatten
Reshape = keras.layers.Reshape


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
# t = np.expand_dims(np.array(x_test), axis=3)

print(x_train.shape)
print(x_test.shape)


input_layer = Input(shape=(28, 28, 1))
x = Flatten()(input_layer)
x = Dense(128, activation="relu")(x)
x = Dense(64, activation="relu")(x)

encoded = Dense(20, activation="relu")(x)

d = Dense(64, activation="relu")(encoded)
d = Dense(28 * 28, activation="relu")(d)
decoded = Reshape((28, 28, 1))(d)

model = Model(input_layer, decoded, name="autoencoder")
model.compile(optimizer="adam", loss="mse")

batch_size = 100

history = model.fit(x_train, x_train, batch_size=batch_size, epochs=20)


n = 10
decoded_imgs = model.predict(x_test[:n])
fig, axes = plt.subplots(2, n, figsize=(15, 6))

for i in range(n):
    axes[0, i].imshow(x_test[i].reshape(28, 28), cmap="gray")
    axes[1, i].imshow(decoded_imgs[i].reshape(28, 28), cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].axis("off")

plt.tight_layout()
plt.show()
