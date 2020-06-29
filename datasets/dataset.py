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
        if self.phase =='train' or self.phase=='valid':
            self.root_path = './datas/train/train'
        else:
            self.root_path = './datas/test/test'
        self.tform = Augment(phase=phase, height=380, width=380)


    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        file_name = self.df.loc[idx, 'filename']
        if self.phase == 'train' or self.phase == 'valid':
            target = self.df.loc[idx, 'category']
            image_path = os.path.join(self.root_path, self.num2str(target), file_name)
        else:
            image_path = os.path.join(self.root_path, file_name)

            # img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        img = Image.open(image_path).convert('RGB')
        (w, h) = img.size
        img = np.asarray(img)
        if self.phase=='train':
            if w < 60 or h < 60:
                print('Error image small')
                return self[np.random.choice(len(self.df))]
        img = self.tform.transform(img)

        if self.phase=='train' or self.phase=='valid':
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
