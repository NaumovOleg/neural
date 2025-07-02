from model import SimpleModel
from layers.dense import CustomDense
import keras
import tensorflow as tf

mnist = keras.datasets.mnist
to_categorical = keras.utils.to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, 28 * 28])
x_test = tf.reshape(tf.cast(x_test, tf.float32), [-1, 28 * 28])

y_train = to_categorical(y_train, 10)


model = SimpleModel(
    [CustomDense(64, activation_fn="relu"), CustomDense(10, activation_fn="softmax")],
    epochs=30,
    optimizer=keras.optimizers.Adam(0.001),
)

model.fit(x_train, y_train)


print(model)
