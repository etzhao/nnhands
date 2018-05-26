import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from skimage import io

class HandDataset(Dataset):
    """3D Hand Pose dataset"""

    def __init__(self, csv_file, **kwargs):
        """
        Args:
            csv_file (string): Path to csv file
            transform (callable, optional): Optional transform to be applied
        """
        self.csv = pd.read_csv(csv_file)
        self.transform = kwargs.get('transform',None)
        self.train = kwargs.get('train', True)


    def __len__(self):
        return len(self.csv)



    def __getitem__(self, idx):
        img_name = self.csv.iloc[idx,0]
        image = io.imread(img_name)

        pos_3d = self.csv.iloc[idx, 1:(21*3)+1].as_matrix().astype(float)
        pos_3d = pos_3d.reshape(21,3)

        p18 = pos_3d[17,:]

        pos_3d -= p18

        pos_2d = self.csv.iloc[idx, (21*3)+1:].as_matrix().astype(float)
        pos_2d = pos_2d.reshape(21,2)

        sample = {'image': image,
                  'pos_3d': pos_3d,
                  'pos_2d': pos_2d,
                  'p18': p18}

        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    def __call__(self, sample):
        image = sample['image']
        pos_3d, pos_2d = sample['pos_3d'], sample['pos_2d']
        # swap color axis
        image = image.transpose((2,0,1))
        return {'image': torch.from_numpy(image),
               'pos_3d': torch.from_numpy(pos_3d),
               'pos_2d': torch.from_numpy(pos_2d),
               'p18': torch.from_numpy(p18)}

class Scale(object):
    def __init__(self, h_out, w_out):
        self.h_out = h_out
        self.w_out = w_out

    def __call__(self, sample):
        image = sample['image']
        pos_3d, pos_2d = sample['pos_3d'], sample['pos_2d']
        image = Image.fromarray(image)
        width, height = image.width, image.height
        image = image.resize((self.w_out, self.h_out))
        image = np.array(image).reshape((self.h_out, self.w_out, 3))

        pos_2d = pos_2d * np.array([self.w_out/width, self.h_out/height])

        return {'image': image,
               'pos_3d': pos_3d,
               'pos_2d': pos_2d}
