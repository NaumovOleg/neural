import tensorflow as tf


class CustomDense(tf.Module):
    def __init__(self, units, activation_fn="relu"):
        super().__init__()
        self.units = units
        self.activation_fn = activation_fn
        self.initialized = False
        self.activations = {
            "relu": tf.nn.relu,
            "sigmoid": tf.nn.sigmoid,
            "tanh": tf.nn.tanh,
            "softmax": tf.nn.softmax,
        }

    def __call__(self, x):
        if not self.initialized:
            self.w = tf.Variable(
                tf.random.truncated_normal((x.shape[-1], self.units), stddev=0.1), name="w"
            )
            self.b = tf.Variable(
                tf.zeros([self.units], dtype=tf.float32), name="b"
            )
            self.initialized = True

        if len(x.shape) == 1:
            x = tf.expand_dims(x, axis=0)

        y = x @ self.w + self.b

        return self.activations[self.activation_fn](y)
