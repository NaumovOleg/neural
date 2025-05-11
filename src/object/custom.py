import tensorflow as tf
import keras
from prepare_dataset import VOCDataset
import matplotlib.pyplot as plt
from image_actions import show_image_with_boxes
import numpy as np
from custom_losses import iou_loss, giou_loss_batch

tf.experimental.numpy.experimental_enable_numpy_behavior()

Input = keras.layers.Input
Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D
GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D
Reshape = keras.layers.Reshape
Dense = keras.layers.Dense
Flatten = keras.layers.Flatten
Concatenate = keras.layers.Concatenate
Activation = keras.layers.Activation
Model = keras.models.Model
Huber = keras.losses.Huber
Lambda = keras.layers.Lambda
MaxPooling2D = keras.layers.MaxPooling2D
Huber = keras.losses.Huber
CategoricalAccuracy = keras.metrics.CategoricalAccuracy


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
x = Flatten()(x)
x = Dense(512, activation="relu")(x)
# x = Dense(256, activation="relu")(x)
x = Dense(12)(x)
x = Reshape((2, 6))(x)
bbox_sigmoid = Activation("sigmoid")(x[..., :4])
class_predictions = x[..., 4:]
outputs = Concatenate(axis=-1)([bbox_sigmoid, class_predictions])


model = Model(inputs=inputs, outputs=outputs)
huber_loss = Huber()


@tf.function
def enforce_box_order(boxes):
    x1 = tf.minimum(boxes[..., 0], boxes[..., 2])
    y1 = tf.minimum(boxes[..., 1], boxes[..., 3])
    x2 = tf.maximum(boxes[..., 0], boxes[..., 2])
    y2 = tf.maximum(boxes[..., 1], boxes[..., 3])
    return tf.stack([x1, y1, x2, y2], axis=-1)


@tf.function
def custom_loss(y_true, y_pred):

    true_boxes = enforce_box_order(y_true[..., :4])
    pred_boxes = enforce_box_order(y_pred[..., :4])
    true_classes = y_true[..., 4:]
    pred_classes = y_pred[..., 4:]

    result = giou_loss_batch(true_boxes, pred_boxes)
    iou_losses = tf.reduce_mean(result)
    tf.print("===============", iou_losses)

    # box_loss = huber_loss(true_boxes, pred_boxes)
    class_loss = tf.keras.losses.categorical_crossentropy(
        true_classes, pred_classes, from_logits=True
    )

    return class_loss + 10.0 * iou_losses


model.compile(
    optimizer="adam", loss=custom_loss, metrics=[CategoricalAccuracy(name="cls_acc")]
)

history = model.fit(x_train, y_train, epochs=10)
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
