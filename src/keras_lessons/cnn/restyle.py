import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from utils.image import upload_image, deprocess_img, gram_matrix


source_image, src_img = upload_image("data_sets/car.jpg")
style_image, st_img = upload_image("data_sets/style_3.png")

source_image = keras.applications.vgg19.preprocess_input(
    np.expand_dims(source_image, axis=0)
)
style_image = keras.applications.vgg19.preprocess_input(
    np.expand_dims(style_image, axis=0)
)

vgg = keras.applications.VGG19(include_top=False, weights="imagenet")
vgg.trainable = False

content_layers = ["block5_conv2"]

style_layers = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

style_outputs = [vgg.get_layer(name).output for name in style_layers]
content_outputs = [vgg.get_layer(name).output for name in content_layers]
model_outputs = style_outputs + content_outputs

print(vgg.input)
for m in model_outputs:
    print(m)

model = keras.models.Model(vgg.input, model_outputs)
print(model.summary())
for layer in model.layers:
    layer.trainable = False


def get_feature_representations(model):
    # batch compute content and style features
    styles = model(style_image)
    contents = model(source_image)

    # Get the style and content feature representations from our model
    style_features = [style_layer[0] for style_layer in styles[:num_style_layers]]
    content_features = [
        content_layer[0] for content_layer in contents[num_style_layers:]
    ]
    return style_features, content_features


def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))


def get_style_loss(base_style, gram_target):
    gram_style = gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_style - gram_target))


def compute_loss(
    model, loss_weights, init_image, gram_style_features, content_features
):
    style_weight, content_weight = loss_weights

    outputs = model(init_image)

    style_output_features = outputs[:num_style_layers]
    content_output_features = outputs[num_style_layers:]

    style_score = 0
    content_score = 0

    # Accumulate style losses from all layers
    # Here, we equally weight each contribution of each loss layer
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(
            comb_style[0], target_style
        )

    # Accumulate content losses from all layers
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * get_content_loss(
            comb_content[0], target_content
        )

    style_score *= style_weight
    content_score *= content_weight

    # Get total loss
    loss = style_score + content_score
    return loss, style_score, content_score


num_iterations = 100
content_weight = 1e3
style_weight = 1e-2

style_features, content_features = get_feature_representations(model)
gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

init_image = np.copy(source_image)
init_image = tf.Variable(init_image, dtype=tf.float32)

opt = tf.compat.v1.train.AdamOptimizer(learning_rate=2, beta1=0.99, epsilon=1e-1)
iter_count = 1
best_loss, best_img = float("inf"), None
loss_weights = (style_weight, content_weight)

cfg = {
    "model": model,
    "loss_weights": loss_weights,
    "init_image": init_image,
    "gram_style_features": gram_style_features,
    "content_features": content_features,
}
plt.imshow(src_img)
plt.show()
norm_means = np.array([103.939, 116.779, 123.68])
min_vals = -norm_means
max_vals = 255 - norm_means
imgs = []

for i in range(num_iterations):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)

    loss, style_score, content_score = all_loss
    grads = tape.gradient(loss, init_image)

    opt.apply_gradients([(grads, init_image)])
    clipped = tf.clip_by_value(init_image, min_vals, max_vals)
    init_image.assign(clipped)

    if loss < best_loss:
        # Update best loss and best image from total loss.
        best_loss = loss
        best_img = deprocess_img(init_image.numpy())

        # Use the .numpy() method to get the concrete numpy array
        plot_img = deprocess_img(init_image.numpy())
        imgs.append(plot_img)
        # print("Iteration: {}".format(i))

plt.imshow(best_img)
print(best_loss)
plt.show()
