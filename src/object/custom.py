import tensorflow as tf
import keras
from prepare_dataset import VOCDataset
import matplotlib.pyplot as plt
from image_actions import show_image_with_boxes
import numpy as np

tf.experimental.numpy.experimental_enable_numpy_behavior()

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
Lambda = keras.layers.Lambda
MaxPooling2D = keras.layers.MaxPooling2D
Huber = keras.losses.Huber

DATASET_PATH = "data_sets/object-detecton.v2i.voc/train"
MAX_OBJECTS = 2
CLASSES = 2
dataset = VOCDataset(DATASET_PATH)

x_train, y_boxes, y_labels = dataset.preprocess_dataset(dataset)

y_labels = tf.one_hot(y_labels, depth=2).numpy()


y_train = np.array(
    [[np.concatenate([box, label], axis=1)] for box, label in zip(y_boxes, y_labels)]
).reshape((-1, 2, 6))

print(y_train.shape)


inputs = Input(shape=(224, 224, 3))
x = Conv2D(64, 3, strides=2, padding="same", activation="relu")(inputs)
x = Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
x = Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dense(256, activation="relu")(x)
x = Dense(12)(x)
outputs = Reshape((2, 6))(x)
model = Model(inputs=inputs, outputs=outputs)

huber_loss = Huber()

huber_loss = Huber()


@tf.function
def custom_loss(y_true, y_pred):

    y_true_boxes = y_true[..., :4]
    y_true_classes = y_true[..., 4:]
    pred_boxes = y_pred[..., :4]
    pred_class_logits = y_pred[..., 4:]
    box_loss = huber_loss(y_true_boxes, pred_boxes)
    # class_loss = tf.keras.losses.categorical_crossentropy(
    #     y_true_classes, pred_class_logits, from_logits=True
    # )

    return box_loss


model.compile(optimizer="adam", loss=custom_loss, metrics=["accuracy"])

history = model.fit(x_train, y_train, epochs=3)
to_predict = x_train[0:1]
predicted = model.predict(to_predict)


for bboxes, img in zip(predicted, to_predict):
    boxes = bboxes[:, :4]
    labels = bboxes[:, 4:]
    denormalized = dataset.denormalize_bboxes(boxes)
    show_image_with_boxes(img, denormalized, labels)
