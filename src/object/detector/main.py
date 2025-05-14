from detector import ObjectDetector
from loss import compute_loss
import tensorflow as tf
from object import VOCDataset, show_image_with_boxes
import numpy as np
from helpers import (
    xyxy_to_cxcywh,
    cxcywh_to_xyxy,
    classification_accuracy,
    form_y_train,
)
import matplotlib.pyplot as plt


tf.experimental.numpy.experimental_enable_numpy_behavior()

DATASET_PATH = "data_sets/object-detecton.v2i.voc/train"
MAX_OBJECTS = 2
CLASSES = 2
dataset = VOCDataset(DATASET_PATH)
x_train, y_boxes, y_labels = dataset.preprocess_dataset(dataset)
y_boxes = xyxy_to_cxcywh(y_boxes)
y_labels = tf.one_hot(y_labels, depth=2).numpy()

y_train = form_y_train(y_boxes, y_labels)


model = ObjectDetector(shape=(224, 224, 3))


model.compile(optimizer="adam", loss=compute_loss, metrics=[classification_accuracy])
history = model.fit(x_train, y_train, epochs=15)
to_predict = x_train[3:12]
predicted = model.predict(to_predict)

for bboxes, img in zip(predicted, to_predict):
    boxes = cxcywh_to_xyxy(bboxes[:, :4])
    labels = bboxes[:, 4:]
    denormalized = dataset.denormalize_bboxes(boxes.numpy())
    show_image_with_boxes(img, denormalized, labels)


def show_accuracy_plot():
    """show_accuracy_plot"""
    plt.figure(figsize=(14, 7))
    plt.plot(history.history["loss"], label="loss")
    plt.title("loss")
    plt.show()


show_accuracy_plot()
