import os

import torch
import torchvision.transforms.v2 as v2
from PIL import Image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        sorted_folders = sorted(
            [
                d
                for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d))
            ]
        )

        for label, class_name in enumerate(sorted_folders):
            class_dir = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_dir):
                if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img_path = os.path.join(class_dir, img_name)
                if not os.path.getsize(img_path) > 0:
                    continue

                try:
                    with Image.open(img_path) as img:
                        img.load()
                except Exception:
                    print(f"Bad file: {img_path}")
                    continue

                self.image_paths.append(img_path)
                self.labels.append(label)

        print(f"Number of images: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        if not isinstance(image, torch.Tensor):
            image = v2.functional.to_image(image)

        if image.dtype != torch.float32:
            image = v2.functional.to_dtype(image, dtype=torch.float32, scale=True)

        return image, label
