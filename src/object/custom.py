import tensorflow as tf
import keras
from prepare_dataset import VOCDataset
import matplotlib.pyplot as plt
from image_actions import show_image_with_boxes
import numpy as np
from custom_losses import giou_loss_batch

tf.experimental.numpy.experimental_enable_numpy_behavior()

Input = keras.layers.Input
Conv2D = keras.layers.Conv2D
Reshape = keras.layers.Reshape
Dense = keras.layers.Dense
Flatten = keras.layers.Flatten
Concatenate = keras.layers.Concatenate
Activation = keras.layers.Activation
Model = keras.models.Model
Lambda = keras.layers.Lambda
CategoricalCrossentropy = keras.losses.CategoricalCrossentropy


DATASET_PATH = "data_sets/object-detecton.v2i.voc/train"
MAX_OBJECTS = 2
CLASSES = 2
dataset = VOCDataset(DATASET_PATH)
x_train, y_boxes, y_labels = dataset.preprocess_dataset(dataset)
y_labels = tf.one_hot(y_labels, depth=2).numpy()
y_train = np.array(
    [[np.concatenate([box, label], axis=1)] for box, label in zip(y_boxes, y_labels)]
).reshape((-1, 2, 6))


inputs = Input(shape=(224, 224, 3))
x = Conv2D(64, 3, strides=2, padding="same", activation="relu")(inputs)
x = Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
x = Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
x = Flatten()(x)
x = Dense(512, activation="relu")(x)
x = Dense(12)(x)
x = Reshape((2, 6))(x)
bbox_sigmoid = Activation("sigmoid")(x[..., :4])

class_predictions = x[..., 4:]


def calculate_boxes(box):
    cx = box[..., 0]
    cy = box[..., 1]
    w = tf.maximum(box[..., 2], 0.01)
    h = tf.maximum(box[..., 3], 0.01)

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    return tf.stack([x1, y1, x2, y2], axis=-1)


bbox_output = Lambda(calculate_boxes)(bbox_sigmoid)
outputs = Concatenate(axis=-1)([bbox_output, class_predictions])

model = Model(inputs=inputs, outputs=outputs)
cross_entropy = CategoricalCrossentropy(from_logits=True)


@tf.function
def custom_loss(y_true, y_pred):

    true_boxes = y_true[..., :4]
    pred_boxes = y_pred[..., :4]
    true_classes = y_true[..., 4:6]
    pred_classes = y_pred[..., 4:6]

    result = giou_loss_batch(true_boxes, pred_boxes)
    iou_losses = tf.reduce_mean(result)

    class_loss = cross_entropy(true_classes, pred_classes)

    return class_loss + 8.12 * iou_losses


model.compile(optimizer="adam", loss=custom_loss, metrics=["accuracy"])

history = model.fit(x_train[0:5], y_train[0:5], epochs=7)
to_predict = x_train[0:4]
predicted = model.predict(to_predict)


for bboxes, img in zip(predicted, to_predict):
    boxes = bboxes[:, :4]
    labels = bboxes[:, 4:]
    denormalized = dataset.denormalize_bboxes(boxes)
    show_image_with_boxes(img, denormalized, labels)


def show_accuracy_plot():
    plt.figure(figsize=(14, 7))
    plt.plot(history.history["loss"], label="loss")
    plt.title("loss")
    plt.show()


show_accuracy_plot()
