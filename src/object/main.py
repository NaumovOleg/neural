import tensorflow as tf
from tensorflow.keras import layers, models


def create_model(input_shape=(224, 224, 3), num_classes=2, num_boxes=5):
    inputs = layers.Input(shape=input_shape)

    # Вводная сетка, например, с использованием сверточных слоев
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)

    # Выход для боксов
    bbox_output = layers.Conv2D(
        num_boxes * 4, (1, 1), activation="sigmoid", name="bbox_output"
    )(x)

    # Выход для классов
    class_output = layers.Conv2D(
        num_boxes * num_classes, (1, 1), activation="softmax", name="class_output"
    )(x)

    model = models.Model(inputs=inputs, outputs=[bbox_output, class_output])

    model.compile(
        optimizer="adam",
        loss={"bbox_output": "mse", "class_output": "categorical_crossentropy"},
        metrics={"bbox_output": "mae", "class_output": "accuracy"},
    )

    return model
