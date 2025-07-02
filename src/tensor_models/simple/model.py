import tensorflow as tf
import keras


def cross_entropy(y_real, y_pred):
    return tf.reduce_mean(keras.losses.categorical_crossentropy(y_real, y_pred))


class SimpleModel(tf.Module):
    def __init__(
        self, layers, optimizer=keras.optimizers.Adam(0.001), epochs=500, batch_size=32
    ):
        super().__init__()
        self.layers = layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def predict(self, X):
        return self(X)

    def trainable_variables(self):
        vars = []
        for layer in self.layers:
            vars.extend(layer.trainable_variables)
        return vars

    def fit(self, X_train, y_train):
        dummy_input = tf.expand_dims(X_train[0], axis=0)
        self.predict(dummy_input)

        self.optimizer = type(self.optimizer).from_config(self.optimizer.get_config())
        self.optimizer.build(self.trainable_variables())
        for n in range(self.epochs):
            loss = 0
            for x_batch, y_batch in zip(X_train, y_train):
                with tf.GradientTape() as tape:
                    f_loss = cross_entropy(
                        tf.expand_dims(y_batch, axis=0), self.predict(x_batch)
                    )

                loss += f_loss
                gradients = tape.gradient(f_loss, self.trainable_variables())
                print(gradients)
                self.optimizer.apply_gradients(
                    zip(gradients, self.trainable_variables())
                )

            print(loss.numpy())
