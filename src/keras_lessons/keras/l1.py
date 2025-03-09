import keras

Sequential = keras.models.Sequential
Dense = keras.layers.Dense
Model = keras.models.Model
Input = keras.layers.Input

model = Sequential(
    [
        Input((2,)),
        Dense(1, kernel_regularizer=keras.regularizers.l1(0.01), activation="relu"),
        Dense(1, activation="relu", name="layer2"),
        Dense(1, kernel_regularizer=keras.regularizers.l1(0.01), activation="softmax"),
    ]
)

model2 = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
model_config = model.get_config()
layer_config = model.get_layer("layer2").get_config()

model.trainable = False
model.layers[1].trainable = True

# model_3 = Model([model, Dense(2)])

# print(model_config, end="\n")
# print(layer_config, end="\n")

inputs = Input((2,))
x = Dense(1, activation="relu", name="layer1")(inputs)
x = Dense(1, activation="relu", name="layer2")(x)
x = Dense(1, activation="relu", name="layer3")(x)
output = Dense(1, activation="softmax", name="layer4")(x)

model_4 = Model(inputs=inputs, outputs=output)
