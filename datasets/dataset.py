import os
import numpy as np
import pandas as pd
from PIL import Image
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
            self.root_path = './datas/train/train'
        else:
            self.root_path = './datas/test/test'
        self.tform = Augment(phase=phase, height=224, width=224)


    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        file_name = self.df.loc[idx, 'filename']
        if self.phase == 'train':
            target = self.df.loc[idx, 'category']
            image_path = os.path.join(self.root_path, self.num2str(target), file_name)
        else:
            image_path = os.path.join(self.root_path, file_name)

        # img = cv2.imread(image_path)
        img = Image.open(image_path).convert('RGB')
        (w, h) = img.size
        if self.phase=='train':
            if w < 60 or h < 60:
                return self[idx+1]


        img = self.tform.transform(img)

        if self.phase=='train':
            target = np.array(target)
            target = torch.from_numpy(target)
            return img, target
        else:
            return img


    @staticmethod
    def num2str(x):
        ans = ''
        if(len(str(x))==1):
            ans = '0'+str(x)
        else:
            ans = str(x)
        return ans
