import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import time
import tensorflow.keras.backend as K

Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Input = keras.layers.Input
Reshape = keras.layers.Reshape
BatchNormalization = keras.layers.BatchNormalization
Conv2D = keras.layers.Conv2D
Conv2DTranspose = keras.layers.Conv2DTranspose
Flatten = keras.layers.Flatten
LeakyReLU = keras.layers.LeakyReLU
Lambda = keras.layers.Lambda
Model = keras.models.Model
Sequential = keras.models.Sequential
BinaryCrossentropy = keras.losses.BinaryCrossentropy
Adam = keras.optimizers.Adam

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize the data to be between 0 and 1

x_train = x_train[y_train == 3]
y_train = y_train[y_train == 3]

BUFFER_SIZE = x_train.shape[0]
BATCH_SIZE = 100
BUFFER_SIZE = BUFFER_SIZE // BATCH_SIZE * BATCH_SIZE
HIDEEN_DIM = 2

x_train = x_train[:BUFFER_SIZE]
y_train = y_train[:BUFFER_SIZE]

x_train = x_train / 255
x_test = x_test / 255

x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

train_dataset = (
    tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
)

generator = Sequential(
    [
        Input(shape=(HIDEEN_DIM,)),
        Dense(7 * 7 * 256, activation="relu"),
        BatchNormalization(),
        Reshape((7, 7, 256)),
        Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", activation="relu"),
        BatchNormalization(),
        Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", activation="relu"),
        BatchNormalization(),
        Conv2DTranspose(
            1, (5, 5), strides=(2, 2), padding="same", activation="sigmoid"
        ),
    ]
)

discriminator = Sequential(
    [
        Input(shape=(28, 28, 1)),
        Conv2D(64, (5, 5), strides=(2, 2), padding="same"),
        LeakyReLU(),
        Dropout(0.3),
        Conv2D(128, (5, 5), strides=(2, 2), padding="same"),
        LeakyReLU(),
        Dropout(0.3),
        Flatten(),
        Dense(1),
    ]
)

cross_entropy = BinaryCrossentropy(from_logits=True)


def generator_losses(output):
    return cross_entropy(tf.ones_like(output), output)


def discriminator_losses(real, fake):
    real_loss = cross_entropy(tf.ones_like(real), real)
    fake_loss = cross_entropy(tf.zeros_like(fake), fake)
    return real_loss + fake_loss


generator_optimiser = Adam(learning_rate=1e-4)
discriminator_optimiser = Adam(learning_rate=1e-4)


@tf.function
def train_step(images):
    noise = tf.random.normal((BATCH_SIZE, HIDEEN_DIM))

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_losses = generator_losses(fake_output)
        disc_losses = discriminator_losses(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_losses, generator.trainable_variables
    )
    gradients_of_discriminator = disc_tape.gradient(
        disc_losses, discriminator.trainable_variables
    )
    generator_optimiser.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )
    discriminator_optimiser.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables)
    )

    return gen_losses, disc_losses


def train(dataset, epochs):
    history = []
    MAX_PRINT_LABEL = 10
    th = BUFFER_SIZE // (BATCH_SIZE * MAX_PRINT_LABEL)

    for epoch in range(1, epochs + 1):
        print(f"{epoch}/{EPOCHS}: ", end="")

        start = time.time()
        n = 0

        gen_loss_epoch = 0
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
            gen_loss_epoch += K.mean(gen_loss)
            if n % th == 0:
                print("=", end="")
            n += 1

        history += [gen_loss_epoch / n]
        print(": " + str(history[-1]))
        print("Время эпохи {} составляет {} секунд".format(epoch, time.time() - start))

    return history


# запуск процесса обучения
EPOCHS = 20
history = train(train_dataset, EPOCHS)

plt.plot(history)
plt.grid(True)
plt.show()

# отображение результатов генерации
n = 2
total = 2 * n + 1

plt.figure(figsize=(total, total))

num = 1
for i in range(-n, n + 1):
    for j in range(-n, n + 1):
        ax = plt.subplot(total, total, num)
        num += 1
        img = generator.predict(np.expand_dims([0.5 * i / n, 0.5 * j / n], axis=0))
        plt.imshow(img[0, :, :, 0], cmap="gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

plt.show()
