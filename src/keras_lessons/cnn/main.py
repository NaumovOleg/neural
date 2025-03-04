import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# filters = число  ядер
# kernel_size = размер ядра (в виде кортежа)
# strides = шаг сканирования фильтров  по осям плскости( по умочанию 1 пиесельь)


mnist = keras.datasets.mnist
scaler = MinMaxScaler(feature_range=(0, 1))

Sequential = keras.models.Sequential
Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D
Input = keras.layers.Input
Flatten = keras.layers.Flatten
Dense = keras.layers.Dense

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = np.array(x_test)
x_train = np.array(x_train)
y_test = np.array(y_test)
y_train = np.array(y_train)


scaler.fit(
    np.concatenate((x_train, x_test), axis=0).reshape(
        x_train.shape[0] + x_test.shape[0], -1
    )
)

y_test_categorical = keras.utils.to_categorical(y_test, 10)
y_train_categorical = keras.utils.to_categorical(y_train, 10)

x_train_scaled = scaler.transform(x_train.reshape(x_train.shape[0], -1))
x_test_scaled = scaler.transform(x_test.reshape(x_test.shape[0], -1))

x_train_scaled = x_train_scaled.reshape(x_train.shape)
x_test_scaled = x_test_scaled.reshape(x_test.shape)


# Добавляем  размерность  - канал  ( черно  белый -1 )
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

print(x_test_scaled.shape, x_train.shape)

# x_train_scaled = scaler.transform(x_train.reshape(x_train.shape[0], -1))

model = Sequential(
    [
        Input((28, 28, 1)),
        Conv2D(filters=32, padding="same", kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2), strides=2),
        Conv2D(filters=64, padding="same", kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2), strides=2),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(10, activation="softmax"),
    ]
)


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

hist = model.fit(x_train, y_train_categorical, epochs=5, verbose=1)
model.evaluate(x_test, y_test_categorical)

plt.grid()
plt.plot(hist.history["loss"])
# plt.plot(hist.history["val_loss"])
plt.show()

# predict = model.predict(x_test)
