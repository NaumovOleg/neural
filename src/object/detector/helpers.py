import numpy as np
import tensorflow as tf


def form_y_train(y_boxes, y_labels):
    y_train = []
    for box, label in zip(y_boxes, y_labels):
        combined = np.concatenate([box, label], axis=1)  # (2, 6)
        y_train.append(combined)

    return np.stack(y_train, axis=0)


def enforce_box_order(boxvals):
    x1 = tf.minimum(boxvals[..., 0], boxvals[..., 2])
    y1 = tf.minimum(boxvals[..., 1], boxvals[..., 3])
    x2 = tf.maximum(boxvals[..., 0], boxvals[..., 2])
    y2 = tf.maximum(boxvals[..., 1], boxvals[..., 3])
    return tf.stack([x1, y1, x2, y2], axis=-1)


@tf.function
def xyxy_to_cxcywh(box):
    """Convert bounding box from (x1, y1, x2, y2) to (cx, cy, w, h)."""
    x_min, y_min, x_max, y_max = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    w = x_max - x_min
    h = y_max - y_min
    cx = x_min + w / 2
    cy = y_min + h / 2
    return tf.stack([cx, cy, w, h], axis=-1)


@tf.function
def cxcywh_to_xyxy(box):
    """
    Convert (cx, cy, w, h) to (x_min, y_min, x_max, y_max) using NumPy.
    box: np.ndarray of shape (..., 4)
    """
    cx = box[..., 0]
    cy = box[..., 1]
    w = tf.maximum(box[..., 2], 0.1)
    h = tf.maximum(box[..., 3], 0.1)

    x1 = cx - w / 2.0
    x2 = cx + w / 2.0
    y1 = cy - h / 2.0
    y2 = cy + h / 2.0

    return tf.stack([x1, y1, x2, y2], axis=-1)


def classification_accuracy(y_true, y_pred):
    y_true_classes = tf.argmax(y_true[..., 4:], axis=-1)
    y_pred_classes = tf.argmax(y_pred[..., 4:], axis=-1)
    return tf.reduce_mean(tf.cast(tf.equal(y_true_classes, y_pred_classes), tf.float32))
