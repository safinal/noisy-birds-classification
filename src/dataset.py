import os
import random
import torch
from torchvision import transforms
import os
from PIL import Image
from typing import List
import random

from src.config import batch_size, num_workers, split_ratio

# Base transform for resizing all images to 224x224
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transform for training with augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transform for validation/testing
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Enhanced augmentation for combined dataset
transform_aug = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class Birddataset(torch.utils.data.Dataset):
    def __init__(self, image_dir: str, allowed_classes: List, transform=None, dataset_type: str = None):
        """
        Args:
            image_dir (str): Directory path containing input images.
            allowed_classes (List): List of class names to include.
            transform (callable): Optional transformation to be applied to images.
            dataset_type (str, optional): Type of dataset, e.g., 'Train' or 'Test'. Defaults to 'Train'.
        """
        # Initialize paths and transformation
        self.allowed_classes = allowed_classes
        self.image_dir = image_dir
        self.dataset_type = dataset_type
        self.transform = transform if transform is not None else base_transform
        self.classes = [item for item in os.listdir(self.image_dir) if os.path.isdir(os.path.join(self.image_dir, item))]
        self.samples = []
        
        for class_name in self.classes:
            if class_name in allowed_classes:
                self.images = os.listdir(os.path.join(self.image_dir, class_name))
                for img in self.images:
                    self.samples.append([img, class_name])

        random.seed(87)
        random.shuffle(self.samples)

        if dataset_type == 'Train':
            self.images = self.samples[:int(len(self.samples)*split_ratio)]
        elif dataset_type == 'Test':
            self.images = self.samples[int(len(self.samples)*split_ratio):]
        else:
            self.images = self.samples

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = os.path.join(self.image_dir, self.images[index][1], self.images[index][0])
        image = Image.open(image_path)
        transformed = self.transform(image)
        class_id = self.allowed_classes.index(self.images[index][1])
        return transformed, class_id

def create_train_val_unlabeled_datasets_and_loaders():
    train_dataset = Birddataset(
        image_dir="./Noisy_birds",
        allowed_classes=["budgie", "canary", "duckling", "rubber duck"],
        transform=transform,
        dataset_type='Train',
    )

    val_dataset = Birddataset(
        image_dir="./Noisy_birds",
        allowed_classes=["budgie", "canary", "duckling", "rubber duck"],
        transform=transform_test,
        dataset_type='Test',
    )

    unlabeled_dataset = Birddataset(
        image_dir="./Noisy_birds",
        allowed_classes=["unlabeled"],
        transform=transform_test,  # Use test transform for unlabeled data
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataset, train_loader, val_dataset, val_loader, unlabeled_dataset, unlabeled_loader

class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, labeled_dataset, pseudo_labeled_data, transform=None):
        self.labeled_data = [(labeled_dataset[i][0], labeled_dataset[i][1]) 
                           for i in range(len(labeled_dataset))]
        self.pseudo_data = pseudo_labeled_data
        self.transform = transform if transform is not None else base_transform
        
    def __len__(self):
        return len(self.labeled_data) + len(self.pseudo_data)
    
    def __getitem__(self, idx):
        if idx < len(self.labeled_data):
            return self.labeled_data[idx]
        else:
            pseudo_idx = idx - len(self.labeled_data)
            image, label = self.pseudo_data[pseudo_idx]
            if self.transform:
                image = self.transform(image)
            return image, label

def create_combined_dataset_and_loader(train_dataset, pseudo_labeled_data):
    combined_dataset = CombinedDataset(train_dataset, pseudo_labeled_data, transform_aug)
    combined_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return combined_dataset, combined_loader