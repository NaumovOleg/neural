import tensorflow as tf
from tensorflow.keras import layers, models
import keras
from prepare_dataset import VOCDataset
import matplotlib.pyplot as plt
from image_actions import show_image, show_image_with_boxes

Input = keras.layers.Input
Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D
GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D
Reshape = keras.layers.Reshape
Dense = keras.layers.Dense
Flatten = keras.layers.Flatten
Activation = keras.layers.Activation
Model = keras.models.Model
Huber = keras.losses.Huber

DATASET_PATH = "data_sets/object-detecton.v2i.voc/train"
MAX_OBJECTS = 2
CLASSES = 2
dataset = VOCDataset(DATASET_PATH)


x_train, y_boxes, y_labels = dataset.preprocess_dataset(dataset)


inputs = Input(shape=(224, 224, 3))
x = Conv2D(64, 3, strides=2, padding="same", activation="relu")(inputs)
x = Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
x = Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
x = Flatten()(x)
x = Dense(512, activation="relu")(x)

box_output = Dense(MAX_OBJECTS * 4)(x)
box_output = Reshape((MAX_OBJECTS, 4), name="boxes")(box_output)

class_output = Dense(MAX_OBJECTS * CLASSES)(x)
class_output = Reshape((MAX_OBJECTS, CLASSES))(class_output)
class_output = Activation("softmax", name="labels")(class_output)


model = models.Model(inputs=inputs, outputs=[box_output, class_output])
model.compile(
    optimizer="adam",
    loss={"boxes": Huber(), "labels": "sparse_categorical_crossentropy"},
    metrics={"boxes": "mse", "labels": "accuracy"},
)

history = model.fit(x_train, {"boxes": y_boxes, "labels": y_labels}, epochs=20)

to_predict = x_train[20:27]
predicted_bboxes, predicted_labels = model.predict(to_predict)

for bboxes, img, labels in zip(predicted_bboxes, to_predict, predicted_labels):
    denormalized = dataset.denormalize_bboxes(bboxes)
    show_image_with_boxes(img, denormalized, labels)


def show_accuracy_plot():
    plt.figure(figsize=(14, 7))
    plt.plot(history.history["loss"], label="loss")
    plt.title("loss")
    plt.show()


show_accuracy_plot()
