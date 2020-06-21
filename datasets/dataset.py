import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as albu
from albumentations.pytorch.transforms import ToTensor
import torchvision as tv

def get_transforms(phase, width=512, height=512):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                albu.HorizontalFlip(),
                albu.OneOf([
                    albu.RandomContrast(),
                    albu.RandomGamma(),
                    albu.RandomBrightness(),
                    ], p=0.3),
                albu.OneOf([
                    albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                    albu.GridDistortion(),
                    albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                    ], p=0.3),
                albu.ShiftScaleRotate(),
            ]
        )
    list_transforms.extend(
        [
            albu.Resize(width,height,always_apply=True),
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
            ToTensor(),
        ]
    )
    list_trfms = albu.Compose(list_transforms)
    return list_trfms

class shopeeDataset(Dataset):
    def __init__(self, df, phase='train'):
        super(shopeeDataset, self).__init__()
        self.df = df
        self.df = self.df.reset_index()
        self.phase = phase
        self.root_path = ''
        if self.phase =='train':
            self.root_path = './datas/train'
        else:
            self.root_path = './datas/test'
        self.transforms = get_transforms(phase)


    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        file_name = self.df.loc[idx, 'filename']
        image_path = os.path.join(self.root_path, file_name)
        img = cv2.imread(image_path)
        target = self.df.loc[idx, 'category']
        target = np.array(target)

        augmented = self.transforms(image=img)
        img = augmented['image']
        target = torch.from_numpy(target)

        return img, target

