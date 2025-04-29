import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


def create_model(num_classes):
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(None, None, 3), include_top=False
    )
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    class_output = layers.Dense(num_classes, activation="softmax", name="class_output")(
        x
    )

    # Для предсказания боксов используем 4 координаты для каждого объекта
    bbox_output = layers.Dense(4, activation="sigmoid", name="bbox_output")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=[class_output, bbox_output])

    return model


def custom_loss(y_true, y_pred):
    class_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true[0], y_pred[0])
    bbox_loss = tf.keras.losses.mean_squared_error(y_true[1], y_pred[1])

    # Суммируем потери с некоторыми весами
    total_loss = class_loss + bbox_loss
    return total_loss


model = create_model(num_classes=2)

# Компиляция модели с использованием кастомной функции потерь
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=custom_loss)


def predict_image(image):
    pred_class, pred_bbox = model.predict(np.expand_dims(image, axis=0))

    return pred_class, pred_bbox


# Пример использования
image = plt.imread("path_to_image.jpg")
pred_class, pred_bbox = predict_image(image)

# Отобразим результаты
print(f"Predicted Class: {pred_class}")
print(f"Predicted BBox: {pred_bbox}")
