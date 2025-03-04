import keras
import numpy as np

Vgg = keras.applications.VGG16
image = keras.preprocessing.image

model = Vgg(
    weights="imagenet",
    # include_top=False,
    input_shape=(224, 224, 3),
    classes=1000,
    classifier_activation="softmax",
)

img = image.load_img("data_sets/car.jpg", target_size=(224, 224))
img_array = image.img_to_array(img)

# Expand dimensions to match VGG16 expected shape: (1, 224, 224, 3)
img_array = np.expand_dims(img_array, axis=0)
x = keras.applications.vgg16.preprocess_input(img_array)

print(x.shape)

predicted = model.predict(x)

print(predicted.shape, np.argmax(predicted, axis=1))

decoded_predictions = keras.applications.vgg16.decode_predictions(predicted)[0]

print(decoded_predictions[0][1], decoded_predictions[0][2])
