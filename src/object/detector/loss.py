import tensorflow as tf
from helpers import cxcywh_to_xyxy, enforce_box_order
import keras

CategoricalCrossentropy = keras.losses.CategoricalCrossentropy

tf.experimental.numpy.experimental_enable_numpy_behavior()

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


@tf.function
def giou_loss_single(cxcy_true, cxcy_pred):
    """giou_loss_single"""

    y_true = cxcywh_to_xyxy(cxcy_true)
    y_pred = cxcywh_to_xyxy(cxcy_pred)

    x1 = tf.maximum(y_true[0], y_pred[0])
    y1 = tf.maximum(y_true[1], y_pred[1])
    x2 = tf.minimum(y_true[2], y_pred[2])
    y2 = tf.minimum(y_true[3], y_pred[3])

    inter_area = tf.maximum(x2 - x1, 0.0) * tf.maximum(y2 - y1, 0.0)

    area_true = (y_true[2] - y_true[0]) * (y_true[3] - y_true[1])
    area_pred = (y_pred[2] - y_pred[0]) * (y_pred[3] - y_pred[1])
    union = area_true + area_pred - inter_area
    iou = inter_area / (union + 1e-7)
    xc1 = tf.minimum(y_true[0], y_pred[0])
    yc1 = tf.minimum(y_true[1], y_pred[1])
    xc2 = tf.maximum(y_true[2], y_pred[2])
    yc2 = tf.maximum(y_true[3], y_pred[3])

    c_area = (xc2 - xc1) * (yc2 - yc1)

    giou = iou - (c_area - union) / (c_area + 1e-7)
    return 1.0 - giou


@tf.function
def giou_loss_batch(y_true, y_pred):
    """giou_loss_batch"""

    def loss_per_sample(inputs):
        true_sample, pred_sample = inputs
        losses = tf.map_fn(
            lambda x: giou_loss_single(x[0], x[1]),
            (true_sample, pred_sample),
            fn_output_signature=tf.float32,
        )
        return tf.reduce_mean(losses)

    return tf.map_fn(loss_per_sample, (y_true, y_pred), fn_output_signature=tf.float32)
