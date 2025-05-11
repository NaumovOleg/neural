import tensorflow as tf


@tf.function
def giou_loss_single(y_true, y_pred):
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

    def loss_per_sample(inputs):
        true_sample, pred_sample = inputs
        losses = tf.map_fn(
            lambda x: giou_loss_single(x[0], x[1]),
            (true_sample, pred_sample),
            fn_output_signature=tf.float32,
        )
        return tf.reduce_mean(losses)

    return tf.map_fn(loss_per_sample, (y_true, y_pred), fn_output_signature=tf.float32)
