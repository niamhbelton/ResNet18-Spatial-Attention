import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
from torchvision import transforms
import pdb
from torch.autograd import Variable

class Dataset(data.Dataset):
    def __init__(self, root_dir, task, test=False, transform=None, indexes=None, weights=None):
        super().__init__()
        self.task = task
        self.root_dir = root_dir
        self.test = test
        self.transform = transform

        if self.test == False:
            self.folder_path = self.root_dir + 'train/sagittal/'
            self.folder_path1 = self.root_dir + 'train/axial/'
            self.folder_path2 = self.root_dir + 'train/coronal/'
            self.records = pd.read_csv(
                self.root_dir + 'train-{0}.csv'.format(task), header=None, names=['id', 'label'])
            self.records = self.records.iloc[indexes,:].reset_index(drop=True)

        else:
            self.folder_path = self.root_dir + 'valid/sagittal/'
            self.folder_path1 = self.root_dir + 'valid/axial/'
            self.folder_path2 = self.root_dir + 'valid/coronal/'
            self.records = pd.read_csv(
                self.root_dir + 'valid-{0}.csv'.format(task), header=None, names=['id', 'label'])

        self.records['id'] = self.records['id'].map(
            lambda i: '0' * (4 - len(str(i))) + str(i))
        self.paths = [self.folder_path + filename +
                      '.npy' for filename in self.records['id'].tolist()]
        self.paths1 = [self.folder_path1 + filename +
                      '.npy' for filename in self.records['id'].tolist()]
        self.paths2 = [self.folder_path2 + filename +
                      '.npy' for filename in self.records['id'].tolist()]
        self.labels = self.records['label'].tolist()


        if weights is None:
            pos = np.sum(self.labels)
            neg = len(self.labels) - pos
            self.weights = [1, neg / pos]
        else:
            self.weights = weights

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        array = np.load(self.paths[index])
        array_1 = np.load(self.paths1[index])
        array_2 = np.load(self.paths2[index])

        label = self.labels[index]
        label = torch.FloatTensor([label])

        if self.transform:
            array = self.transform(array)
            array = array.numpy()
            array_1 = self.transform(array_1)
            array_1 = array_1.numpy()
            array_2 = self.transform(array-2)
            array_2 = array_2.numpy()

        array = np.stack((array,)*3)
        array = torch.FloatTensor(array)
        array=array.permute(1,0,2,3)


        array_1 = np.stack((array_1,)*3)
        array_1 = torch.FloatTensor(array_1)
        array_1=array_1.permute(1,0,2,3)


        array_2 = np.stack((array_2,)*3)
        array_2 = torch.FloatTensor(array_2)
        array_2=array_2.permute(1,0,2,3)

        if label.item() == 1:
            weight = np.array([self.weights[1]])
            weight = torch.FloatTensor(weight)
        else:
            weight = np.array([self.weights[0]])
            weight = torch.FloatTensor(weight)

        return array, array_1, array_2, label, weight
