import tensorflow as tf
from tensorflow.keras import layers, models
import keras

Input = keras.layers.Input
MobileNetV2 = keras.applications.MobileNetV2
GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D
Dense = keras.layers.Dense
Model = keras.models.Model

input_shape = (224, 224, 3)
num_classes = 2

base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
base_model.trainable = False

inputs = Input(shape=input_shape)
x = base_model(inputs)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)

bbox_output = Dense(4, name="bbox")(x)
class_output = Dense(num_classes, activation="softmax", name="class")(x)

model = Model(inputs=inputs, outputs=[bbox_output, class_output])


model.compile(
    optimizer="adam",
    loss={
        "bbox": "mean_squared_error",  # для координат
        "class": "categorical_crossentropy",  # для классов
    },
    metrics={"class": "accuracy"},
)

# model.fit(
#     X_train,
#     {"bbox_output": y_train_bbox, "class_output": y_train_class},
#     epochs=10,
#     batch_size=16,
# )

# Предполагается, что у тебя есть `train_dataset` и `val_dataset`
model.fit(train_dataset, validation_data=val_dataset, epochs=50)


y_train = [
    [
        [10, 20, 33, 44],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, -1, -1, -1],
    ],
    [
        [10, 20, 33, 44],
        [44, 33, 12, 3],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, -1, -1],
    ],
]
