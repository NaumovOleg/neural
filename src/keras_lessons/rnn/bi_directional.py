import keras
import numpy as np
import matplotlib.pyplot as plt

Sequential = keras.models.Sequential
GRU = keras.layers.GRU
Dense = keras.layers.Dense
Bidirectional = keras.layers.Bidirectional
Input = keras.layers.Input
Adam = keras.optimizers.Adam

rng = np.random.default_rng(42)


N = 10000
off = 3
length = off * 2 + 1
data = np.array([np.sin(x / 20) for x in range(N)]) + 0.1 * rng.standard_normal(N)
off = 3
length = off * 2 + 1

X = np.array(
    [
        np.diag(np.hstack((data[i : i + off], data[i + off + 1 : i + length])))
        for i in range(N - length)
    ]
)
Y = data[off : N - off - 1]

# M = 10
# XX = np.zeros(M)
# XX[:off] = data[:off]
# for i in range(M - off - 1):
#     x = np.diag(np.hstack((XX[i : i + off], data[i + off + 1 : i + length])))
#     x = np.expand_dims(x, axis=0)
#     print(x.shape)

model = Sequential(
    [
        Input((length - 1, length - 1)),
        Bidirectional(GRU(2)),
        Dense(1, activation="linear"),
    ]
)
optimizer = Adam(learning_rate=0.01)
model.compile(loss="mse", optimizer=optimizer, metrics=["mse", "accuracy"])

history = model.fit(X, Y, epochs=10)

print(model.summary())

M = 100
XX = np.zeros(M)
XX[:off] = data[:off]
for i in range(M - off - 1):
    x = np.diag(np.hstack((XX[i : i + off], data[i + off + 1 : i + length])))
    x = np.expand_dims(x, axis=0)
    y = model.predict(x)
    print("-----------", y)
    XX[i + off] = y

plt.plot(XX[:M])
plt.plot(data[:M])
plt.show()
