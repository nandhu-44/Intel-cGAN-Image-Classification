import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class IntelDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = os.path.join(root, f'seg_{split}', f'seg_{split}')
        self.transform = transform
        self.classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        self.labels = []
        print(f"Loading {split} dataset...")
        for cls in self.classes:
            cls_path = os.path.join(self.root, cls)
            try:
                for img_name in os.listdir(cls_path):
                    self.images.append(os.path.join(cls_path, img_name))
                    self.labels.append(self.class_to_idx[cls])
            except FileNotFoundError:
                print(f"Warning: Class directory {cls_path} not found")
        print(f"Found {len(self.images)} images in {split} dataset")

    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)