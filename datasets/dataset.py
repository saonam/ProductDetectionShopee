import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
import torchvision as tv
from datasets import Augment


class shopeeDataset(Dataset):
    def __init__(self, df, phase='train'):
        super(shopeeDataset, self).__init__()
        self.df = df
        self.df = self.df.reset_index()
        self.phase = phase
        self.root_path = ''
        if self.phase =='train':
            self.root_path = './data/train/train'
        else:
            self.root_path = './test/test'
        self.tform = Augment(phase=phase)


    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        target = self.df.loc[idx, 'category']
        file_name = self.df.loc[idx, 'filename']
        image_path = os.path.join(self.root_path, self.num2str(target), file_name)
        img = cv2.imread(image_path)

        target = np.array(target)

        img = self.tform.transform(image=img)
        target = torch.from_numpy(target)

        return img, target
    @staticmethod
    def num2str(x):
        ans = ''
        if(len(str(x))==1):
            ans = '0'+str(x)
        else:
            ans = str(x)
        return ans
