import keras
import numpy as np
import tensorflow as tf

image = keras.preprocessing.image


def upload_image(path: str, target_size=(224, 224)):
    """
    Upload an image from the given path and convert it to a numpy array
    Args: path (str): the path to the image
    Returns: numpy.array: the uploaded image as a numpy array
    """
    img = image.load_img(path, target_size=target_size)
    return (image.img_to_array(img), img)


def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, (
        "Input to deprocess image must be an image of "
        "dimension [1, height, width, channel] or [height, width, channel]"
    )
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # perform the inverse of the preprocessing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype("uint8")
    return x


def gram_matrix(input_tensor):
    # We make the image channels first
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)
