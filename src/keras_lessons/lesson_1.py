import keras
import matplotlib.pyplot as plt
import numpy as np
from data_sets import temperature
from keras import Sequential

celsium, fahrenheit = temperature

optimiser = keras.optimizers.Adam(learning_rate=0.1)
Dense = keras.layers.Dense


model = Sequential()
model.add(Dense(units=1, input_shape=(1,), activation="linear"))
model.compile(optimizer=optimiser, loss="mse")
history = model.fit(celsium, fahrenheit, epochs=500, verbose=0)


# plt.plot(history.history["loss"])
# plt.grid(True)
# plt.show()

predicted = model.predict(np.array([100]))
weights = model.get_weights()
