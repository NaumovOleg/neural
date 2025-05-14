import tensorflow as tf
import keras


Model = keras.models.Model
Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D
Flatten = keras.layers.Flatten
Dense = keras.layers.Dense
Input = keras.layers.Input
InputLayer = keras.layers.InputLayer
Dropout = keras.layers.Dropout
Activation = keras.layers.Activation
CategoricalCrossentropy = keras.losses.CategoricalCrossentropy
Concatenate = keras.layers.Concatenate


class ObjectDetector(Model):
    """Detector model for object detection."""

    def __init__(self, shape=(224, 224, 3)):
        super().__init__()
        self.shape = shape
        self.conv1 = Conv2D(64, 3, strides=2, padding="same", activation="relu")
        self.conv2 = Conv2D(128, 3, strides=2, padding="same", activation="relu")
        self.conv3 = Conv2D(256, 3, strides=2, padding="same", activation="relu")
        self.flatten = Flatten()
        self.dense1 = Dense(512, activation="relu")
        self.dense2 = Dense(12)
        self.dropout = Dropout(0.2)
        self.sigmoid_activation = Activation("sigmoid")
        self.softmax_activation = Activation("softmax")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        flatten = self.flatten(x)
        x = self.dense1(flatten)
        x = self.dropout(x)
        x = self.dense2(x)
        x = tf.reshape(x, (-1, 2, 6))
        bbox_output = self.sigmoid_activation(x[..., :4])
        class_predictions = self.softmax_activation(x[..., 4:])
        return tf.concat([bbox_output, class_predictions], axis=-1)
