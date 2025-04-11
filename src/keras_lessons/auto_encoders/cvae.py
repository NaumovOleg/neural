import keras
import numpy as np
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt


Dense = keras.layers.Dense
Input = keras.layers.Input
Reshape = keras.layers.Reshape
Flatten = keras.layers.Flatten
Input = keras.layers.Input
Lambda = keras.layers.Lambda
BatchNormalization = keras.layers.BatchNormalization
Dropout = keras.layers.Dropout
concatenate = keras.layers.concatenate
Model = keras.models.Model


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255

x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

HIDDEN_DIM = 2
BATCH_SIZE = 100
NUM_CLASSES = 10

y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test_cat = keras.utils.to_categorical(y_test, NUM_CLASSES)


def dropout_and_batch(x):
    return Dropout(0.3)(BatchNormalization()(x))


input_img = Input(shape=(28, 28, 1))
fl = Flatten()(input_img)
lb = Input(shape=(NUM_CLASSES,))
x = concatenate([fl, lb])
x = Dense(26, activation="relu")(x)
x = dropout_and_batch(x)
x = Dense(128, activation="relu")(x)
x = dropout_and_batch(x)

z_means = Dense(HIDDEN_DIM)(x)
z_log_var = Dense(HIDDEN_DIM)(x)


# Генератор  случайных величин
def noizer(args):
    global z_mean, z_log_var
    z_mean, z_log_var = args
    N = tf.keras.backend.random_normal(
        shape=(BATCH_SIZE, HIDDEN_DIM), mean=0.0, stddev=1.0
    )
    return tf.exp(z_log_var / 2) * N + z_mean


h = Lambda(noizer, output_shape=(HIDDEN_DIM,))([z_means, z_log_var])

input_dec = Input(shape=(HIDDEN_DIM,))
lb_dec = Input(shape=(NUM_CLASSES,))
d = concatenate([input_dec, lb_dec])
d = Dense(128, activation="elu")(d)
d = dropout_and_batch(d)
d = Dense(256, activation="elu")(d)
d = dropout_and_batch(d)
d = Dense(28 * 28, activation="sigmoid")(d)
decoded = Reshape((28, 28, 1))(d)

encoder = Model([input_img, lb], h, name="encoder")
decoder = Model([input_dec, lb_dec], decoded, name="decoder")

cvae = Model(
    [input_img, lb, lb_dec], decoder([encoder([input_img, lb]), lb_dec]), name="cvae"
)


def vae_loss(x, y):
    x = tf.reshape(x, shape=(BATCH_SIZE, 28 * 28))
    y = tf.reshape(y, shape=(BATCH_SIZE, 28 * 28))
    loss = tf.reduce_sum(tf.square(x - y), axis=-1)
    kl_loss = -0.5 * tf.reduce_sum(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1
    )
    return (loss + kl_loss) / 2 / 28 / 28


z_meaner = keras.Model([input_img, lb], z_means)
tr_style = keras.Model(
    [input_img, lb, lb_dec],
    decoder([z_meaner([input_img, lb]), lb_dec]),
    name="tr_style",
)

cvae.compile(optimizer="adam", loss=vae_loss)
cvae.fit(
    [x_train, y_train_cat, y_train_cat],
    x_train,
    epochs=5,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


lb = lb_dec = y_test_cat
predictions = cvae.predict([x_test, lb, lb_dec], batch_size=BATCH_SIZE)


plt.scatter(predictions[:, 0], predictions[:, 1])
plt.show()


n = 4
total = 2 * n + 1
input_lbl = np.zeros((1, NUM_CLASSES))
input_lbl[0, 5] = 1

plt.figure(figsize=(total, total))

h = np.zeros((1, HIDDEN_DIM))
num = 1
for i in range(-n, n + 1):
    for j in range(-n, n + 1):
        ax = plt.subplot(total, total, num)
        num += 1
        h[0, :] = [1 * i / n, 1 * j / n]
        img = decoder.predict([h, input_lbl])
        plt.imshow(img.squeeze(), cmap="gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


def plot_digits(*images):
    images = [x.squeeze() for x in images]
    n = min([x.shape[0] for x in images])

    plt.figure(figsize=(n, len(images)))
    for j in range(n):
        for i in range(len(images)):
            ax = plt.subplot(len(images), n, i * n + j + 1)
            plt.imshow(images[i][j])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.show()


dig1 = 5
dig2 = 2

num = 10
X = x_train[y_train == dig1][:num]

lb_1 = np.zeros((num, NUM_CLASSES))
lb_1[:, dig1] = 1

plot_digits(X)

for i in range(NUM_CLASSES):
    lb_2 = np.zeros((num, NUM_CLASSES))
    lb_2[:, i] = 1

    Y = tr_style.predict([X, lb_1, lb_2], batch_size=num)
    plot_digits(Y)
