import keras

Sequential = keras.models.Sequential
Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D
Flatten = keras.layers.Flatten
Dense = keras.layers.Dense

train_images = keras.datasets.mnist.load_data()[0][0]
train_labels = keras.datasets.mnist.load_data()[1]
test_images, test_labels = (
    keras.datasets.mnist.load_data()[0][0],
    keras.datasets.mnist.load_data()[1],
)
