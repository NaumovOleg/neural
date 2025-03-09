import keras
import matplotlib.pyplot as plt

Input = keras.layers.Input
Dense = keras.layers.Dense
Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D
Flatten = keras.layers.Flatten
Dropout = keras.layers.Dropout
Sequential = keras.models.Sequential
Conv2D = keras.layers.Conv2D
Sequential = keras.models.Sequential


model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(7, activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
