import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def load_data(image_paths, annotation_paths, input_size=(224, 224)):
    images = []
    bboxes = []
    classes = []

    for img_path, ann_path in zip(image_paths, annotation_paths):
        # Загрузка изображения
        img = load_img(img_path, target_size=input_size)
        img = img_to_array(img)
        images.append(img)

        # Получение аннотаций
        objects = parse_annotation(ann_path)
        bbox = []
        class_ = []

        for obj in objects:
            class_name, xmin, ymin, xmax, ymax = obj
            bbox.extend([xmin, ymin, xmax, ymax])
            if class_name == "карандаш":
                class_.append(0)  # 0 - карандаш
            elif class_name == "резинка":
                class_.append(1)  # 1 - резинка

        # Если объектов нет, добавляем пустые значения
        while len(bbox) < 20:
            bbox.append(0)
            class_.append(-1)

        bboxes.append(bbox)
        classes.append(class_)

    return np.array(images), np.array(bboxes), np.array(classes)


def create_model(input_shape=(224, 224, 3), num_classes=2, num_boxes=5):
    inputs = layers.Input(shape=input_shape)

    # Вводная сетка, например, с использованием сверточных слоев
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)

    # Выход для боксов
    bbox_output = layers.Conv2D(
        num_boxes * 4, (1, 1), activation="sigmoid", name="bbox_output"
    )(x)

    # Выход для классов
    class_output = layers.Conv2D(
        num_boxes * num_classes, (1, 1), activation="softmax", name="class_output"
    )(x)

    model = models.Model(inputs=inputs, outputs=[bbox_output, class_output])

    model.compile(
        optimizer="adam",
        loss={"bbox_output": "mse", "class_output": "categorical_crossentropy"},
        metrics={"bbox_output": "mae", "class_output": "accuracy"},
    )

    return model
