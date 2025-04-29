import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

d = np.array([])


CLASS_MAP = {"pencil": 0, "eraser": 1}


class VOCDataset(Dataset):
    def __init__(self, root_dir, image_size=(224, 224), transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.images = [f for f in os.listdir(root_dir) if f.endswith(".jpg")]
        self.w = image_size[0]
        self.h = image_size[1]

    def __len__(self):
        return len(self.images)

    def normalize_bboxes(self, bboxes: np.ndarray):
        bboxes[:, [0, 2]] /= self.w  # x_min, x_max
        bboxes[:, [1, 3]] /= self.h
        return bboxes

    def denormalize_bboxes(self, bboxes: np.ndarray):
        bboxes[:, [0, 2]] *= self.w  # x_min, x_max
        bboxes[:, [1, 3]] *= self.h
        return bboxes

    def __getitem__(self, idx):
        img_filename = self.images[idx]
        img_path = os.path.join(self.root_dir, img_filename)
        xml_path = img_path.replace(".jpg", ".xml")

        # Чтение изображения
        image = cv2.imread(img_path)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Переводим BGR -> RGB

        # Чтение аннотаций
        boxes = []
        labels = []

        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            label = obj.find("name").text
            label_id = CLASS_MAP[label]

            bndbox = obj.find("bndbox")
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label_id)

        # Переводим в тензоры
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = transformed["image"]
            target["boxes"] = torch.tensor(transformed["bboxes"])

        # Перевести изображение в формат float32 и нормализовать в диапазон [0,1]
        image = torch.as_tensor(image, dtype=torch.float32) / 255.0
        # image = image.permute(2, 0, 1)  # Из HWC в CHW формат для PyTorch

        return (image, target)

    def preprocess_dataset(self, dataset):
        x_train = []
        y_bboxes = []
        y_labels = []

        for image_tensor, target in dataset:

            boxes = self.normalize_bboxes(np.array(target["boxes"]))
            labels = np.array(target["labels"])

            # добавляем в массивы
            x_train.append(image_tensor)
            y_bboxes.append(boxes)
            y_labels.append(labels)

        return (
            np.array(x_train, dtype=np.float32),
            np.array(y_bboxes, dtype=np.float32),
            np.array(y_labels, dtype=np.int32),
        )
