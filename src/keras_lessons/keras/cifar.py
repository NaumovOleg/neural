import keras
import matplotlib.pyplot as plt
from sklearn import model_selection

cifar10 = keras.datasets.cifar10
Sequential = keras.models.Sequential
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Conv2D = keras.layers.Conv2D
Input = keras.layers.Input
MaxPooling2D = keras.layers.MaxPooling2D
Flatten = keras.layers.Flatten


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

x_train, x_validate, y_train, y_validate = model_selection.train_test_split(
    x_train, y_train, test_size=0.2
)


def sum():
    # Function to return the sum of 1 and 2
    return 1 + 2


print(x_train.shape, y_train.shape)

params = {"kernel_size": (3, 3), "activation": "relu", "padding": "same"}

model = Sequential(
    [
        Input((32, 32, 3)),
        Conv2D(filters=32, **params),
        Conv2D(filters=32, **params),
        MaxPooling2D(2),
        Conv2D(filters=64, **params),
        Conv2D(filters=64, **params),
        MaxPooling2D(2),
        Conv2D(filters=128, **params),
        Conv2D(filters=128, **params),
        Conv2D(filters=128, **params),
        Conv2D(filters=128, **params),
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# hist = model.fit(
#     x_train,
#     y_train,
#     epochs=10,
#     verbose=1,
#     validation_split=0.2,
#     validation_data=(x_validate, y_validate),
# )
# evaluates = model.evaluate(x_test, y_test)

# print(evaluates)


# plt.grid()
# plt.plot(hist.history["loss"])
# plt.plot(hist.history["val_loss"])
# plt.show()
