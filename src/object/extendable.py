import tensorflow as tf
import matplotlib.pyplot as plt
from image_actions import show_image_with_boxes
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Flatten,
    Activation,
    Lambda,
    Concatenate,
    Dropout,
)
from tensorflow.keras.models import Model
from tensorflow.keras.losses import CategoricalCrossentropy
from custom_losses import giou_loss_batch
from prepare_dataset import VOCDataset
import numpy as np

tf.experimental.numpy.experimental_enable_numpy_behavior()

DATASET_PATH = "data_sets/object-detecton.v2i.voc/train"
MAX_OBJECTS = 2
CLASSES = 2
dataset = VOCDataset(DATASET_PATH)
x_train, y_boxes, y_labels = dataset.preprocess_dataset(dataset)
y_labels = tf.one_hot(y_labels, depth=2).numpy()
y_train = np.array(
    [[np.concatenate([box, label], axis=1)] for box, label in zip(y_boxes, y_labels)]
).reshape((-1, 2, 6))


class ObjectDetector(Model):
    """Object detector"""

    def __init__(self):
        super().__init__()
        self.convv1 = Conv2D(
            64, 3, strides=2, padding="same", activation="relu", name="ConvV1"
        )
        self.convv2 = Conv2D(
            128, 3, strides=2, padding="same", activation="relu", name="ConvV2"
        )
        self.convv3 = Conv2D(
            256, 3, strides=2, padding="same", activation="relu", name="ConvV3"
        )
        self.flatten = Flatten(name="Flatten")
        self.dense1 = Dense(512, activation="relu", name="Dense1")
        self.dense2 = Dense(12, name="Dense2")
        self.bbox_output = Lambda(self.calculate_boxes)
        self.outputs = Concatenate(axis=-1)
        self.dropout = Dropout(0.2)
        self.bbox_sigmoid = Activation("sigmoid")
        self.cross_entropy = CategoricalCrossentropy(from_logits=True)

    def call(self, inputs):
        """call"""
        x = self.convv1(inputs)
        x = self.convv2(x)
        x = self.convv3(x)
        flatten = self.flatten(x)
        x = self.dense1(flatten)
        x = self.dropout(x)
        x = self.dense2(x)
        x = tf.reshape(x, (-1, 2, 6))
        bbox_sigmoid = self.bbox_sigmoid(x[..., :4])
        class_predictions = x[..., 4:]
        bbox_output = self.bbox_output(bbox_sigmoid)
        return tf.concat([bbox_output, class_predictions], axis=-1)

    @staticmethod
    def calculate_boxes(box):
        """calculate_boxes"""
        cx = box[..., 0]
        cy = box[..., 1]
        w = tf.maximum(box[..., 2], 0.01)
        h = tf.maximum(box[..., 3], 0.01)
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return tf.stack([x1, y1, x2, y2], axis=-1)


cross_entropy = CategoricalCrossentropy(from_logits=True)


@tf.function
def compute_loss(y_true, y_pred):
    """compute_loss"""
    true_boxes = y_true[..., :4]
    pred_boxes = y_pred[..., :4]
    true_classes = y_true[..., 4:6]
    pred_classes = y_pred[..., 4:6]

    result = giou_loss_batch(true_boxes, pred_boxes)
    iou_losses = tf.reduce_mean(result)
    class_loss = cross_entropy(true_classes, pred_classes)

    return class_loss + 9.12 * iou_losses


model = ObjectDetector()

model.compile(optimizer="adam", loss=compute_loss, metrics=["accuracy"])
history = model.fit(x_train, y_train, epochs=15)
to_predict = x_train[3:12]
predicted = model.predict(to_predict)

for bboxes, img in zip(predicted, to_predict):
    boxes = bboxes[:, :4]
    labels = bboxes[:, 4:]
    denormalized = dataset.denormalize_bboxes(boxes)
    show_image_with_boxes(img, denormalized, labels)


def show_accuracy_plot():
    """show_accuracy_plot"""
    plt.figure(figsize=(14, 7))
    plt.plot(history.history["loss"], label="loss")
    plt.title("loss")
    plt.show()


show_accuracy_plot()
