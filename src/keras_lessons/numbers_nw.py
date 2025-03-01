import keras
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mnist = keras.datasets.mnist
Dense = keras.layers.Dense
Flatten = keras.layers.Flatten
Sequential = keras.models.Sequential
Input = keras.layers.Input
Dropout = keras.layers.Dropout
BatchNormalization = keras.layers.BatchNormalization


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
# model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(10, activation="softmax"))

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)

x_train, x_val, y_train, y_val = train_test_split(
    x_train_scaled, y_train_cat, test_size=0.2
)

hist = model.fit(
    x_train,
    y_train,
    batch_size=30,
    epochs=5,
    verbose=1,
    validation_data=(x_val, y_val),
)

model.evaluate(x_test_scaled, y_test_cat)

predicted = model.predict(x_test_scaled)
predicted = np.argmax(predicted, axis=1)

not_valid_mask = predicted == y_test
not_valid = y_test[~(predicted == y_test)]

plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.show()
