import keras
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

Sequential = keras.models.Sequential
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Input = keras.layers.Input
Adam = keras.optimizers.Adam
BatchNormalization = keras.layers.BatchNormalization
Flatten = keras.layers.Flatten

dataset = pd.read_csv("data_sets/csv/football_matches.csv")
dataset.drop_duplicates(inplace=True)
dataset.reset_index(inplace=True)
dataset.drop(["season", "date", "X"], axis=1, inplace=True)

team_encoder = LabelEncoder()
result_laber_encoder = LabelEncoder()
result_laber_encoder.fit(dataset["result"].unique())
team_encoder.fit(dataset["home_team"].unique())
dataset["home_team"] = team_encoder.transform(dataset["home_team"])
dataset["away_team"] = team_encoder.transform(dataset["away_team"])
dataset["result"] = result_laber_encoder.transform(dataset["result"])

X_train = dataset.drop("result", axis=1)
y_train = dataset.result
x_train, x_test, y_train, y_test = train_test_split(X_train, y_train, random_state=42)

y_train = keras.utils.to_categorical(y_train, 3)
x_train, x_validate, y_train, y_validate = train_test_split(
    x_train, y_train, random_state=42, test_size=0.2
)


def transform_team(team_val: int | str, encoder):
    if isinstance(team_val, str):
        return encoder.fit_transform([team_val])
    return encoder.inverse_transform([team_val])


# y_train = np.array(y_train).reshape(-1, 1)
print("=========", x_train.shape, y_train.shape)

optimiser = Adam(learning_rate=0.007)

model = Sequential(
    [
        Input(shape=(x_train.shape[1],)),
        # Flatten(),
        Dense(30, activation="relu"),
        # Dropout(0.1),
        # BatchNormalization(),
        Dense(30, activation="relu"),
        Dense(3, activation="softmax"),
    ]
)

# model.compile(
#     optimizer=optimiser, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
# )
model.compile(
    optimizer=optimiser, loss="categorical_crossentropy", metrics=["accuracy"]
)

hist = model.fit(
    x_train,
    y_train,
    # batch_size=30,
    epochs=30,
    verbose=1,
    validation_data=(x_validate, y_validate),
)

plt.grid()
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.show()

predicted = model.predict(x_test)
predicted = np.argmax(predicted, axis=1)

raw_length_end = 20
raw_length_start = 0

batch_predicted = predicted[raw_length_start:raw_length_end]
batch_test = np.array(y_test[raw_length_start:raw_length_end])

invalid = batch_test != batch_predicted

print("test-----", batch_predicted, end="\n\n")
print("predicted", batch_test, end="\n\n")
print("loss", hist.history["loss"][-1], end="\n\n")
print("invalid", len(y_test), np.array(y_test[~(predicted == y_test)]), end="\n\n")
