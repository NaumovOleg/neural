import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def show_image(image, bbox):
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min

    # Create figure and axes
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Draw the rectangle
    rect = patches.Rectangle(
        (x_min, y_min), width, height, linewidth=2, edgecolor="r", facecolor="none"
    )
    ax.add_patch(rect)

    # Optionally annotate coordinates
    ax.text(x_min, y_min - 5, f"({x_min}, {y_min})", color="red")

    # Show
    plt.axis("off")
    plt.show()


def show_image_with_boxes(image, bboxes, class_ids):

    class_names = ["pencil", "eraser"]
    class_colors = ["blue", "green"]

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for bbox, classes in zip(bboxes, class_ids):
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min

        color = class_colors[np.argmax(classes)]
        label = class_names[np.argmax(classes)]

        rect = patches.Rectangle(
            (x_min, y_min),
            width,
            height,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x_min, y_min - 5, label, color=color, fontsize=10, backgroundcolor="white"
        )

    plt.axis("off")
    plt.show()
