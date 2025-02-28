import keras
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

mnist = keras.datasets.mnist
Dense = keras.layers.Dense
Flatten = keras.layers.Flatten
Sequential = keras.models.Sequential
Input = keras.layers.Input

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

scaler = MinMaxScaler(feature_range=(0, 1))
categorical_encoder = OneHotEncoder()

x_train_scaled = scaler.fit_transform(x_train.reshape(x_train.shape[0], -1))
x_test_scaled = scaler.fit_transform(x_test.reshape(x_test.shape[0], -1))

x_train_scaled = x_train_scaled.reshape(x_train.shape)
x_test_scaled = x_test_scaled.reshape(x_test.shape)


model = Sequential()
model.add(Input((28, 28, 1)))
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dense(10, activation="softmax"))

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(
    x_train_scaled,
    y_train_cat,
    batch_size=32,
    epochs=5,
    verbose=1,
    validation_split=0.2,
)

model.evaluate(x_test_scaled, y_test_cat)

predicted = model.predict(x_test_scaled)
predicted = np.argmax(predicted, axis=1)
print(predicted, end="\n")


not_valid_mask = predicted == y_test
not_valid = y_test[~(predicted == y_test)]

print(not_valid, end="\n")
