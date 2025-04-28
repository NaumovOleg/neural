import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
import cv2

# Словарь классов
CLASS_MAP = {"pencil": 0, "eraser": 1}

# Путь к твоему датасету
DATASET_PATH = "data_sets/object-detecton.v2i.voc/train"


class VOCDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.images = [f for f in os.listdir(root_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_filename = self.images[idx]
        img_path = os.path.join(self.root_dir, img_filename)
        xml_path = img_path.replace(".jpg", ".xml")

        # Чтение изображения
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Переводим BGR -> RGB

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
        image = image.permute(2, 0, 1)  # Из HWC в CHW формат для PyTorch

        return image, target


# Пример использования:
dataset = VOCDataset(root_dir=DATASET_PATH)

# Проверка первого элемента
image, target = dataset[0]
print(f"Image shape: {image.shape}")
print(f"Target: {target}")
