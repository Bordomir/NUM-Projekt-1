from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import torchvision.transforms.v2 as v2


class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for label, class_name in enumerate(os.listdir(data_dir)):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(label)

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