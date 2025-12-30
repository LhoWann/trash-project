import os
import torch
import numpy as np
import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import config

class GarbageDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.data_dir = config.DATA_DIR
        self.batch_size = config.BATCH_SIZE
        self.num_workers = config.NUM_WORKERS
        self.pin_memory = config.PIN_MEMORY
        self.class_weights = None
        self.class_names = None

    def setup(self, stage=None):
        train_transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.train_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform=train_transform)
        self.val_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'val'), transform=val_transform)
        
        test_path = os.path.join(self.data_dir, 'test')
        if os.path.exists(test_path):
            self.test_dataset = datasets.ImageFolder(test_path, transform=val_transform)
        else:
            self.test_dataset = self.val_dataset

        self.class_names = self.train_dataset.classes
        
        targets = self.train_dataset.targets
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(targets),
            y=targets
        )
        self.class_weights = torch.tensor(class_weights, dtype=torch.float)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, pin_memory=self.pin_memory)