import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
import pdb
from torch.autograd import Variable

class Dataset(data.Dataset):
    def __init__(self, root_dir, task, plane, test=False, transform=None, indexes=None, weights=None ):
        super().__init__()
        self.task = task
        self.plane = plane
        self.root_dir = root_dir
        self.test=test
        if self.test == False:
            self.folder_path = self.root_dir + 'train/{0}/'.format(plane)
            self.records = pd.read_csv(
                self.root_dir + 'train-{0}.csv'.format(task), header=None, names=['id', 'label'])
            self.records = self.records.iloc[indexes,:].reset_index(drop=True)
        else:
            self.folder_path = self.root_dir + 'valid/{0}/'.format(plane)

            self.records = pd.read_csv(
                self.root_dir + 'valid-{0}.csv'.format(task), header=None, names=['id', 'label'])

        self.records['id'] = self.records['id'].map(
            lambda i: '0' * (4 - len(str(i))) + str(i))
        self.paths = [self.folder_path + filename +
                      '.npy' for filename in self.records['id'].tolist()]
        self.labels = self.records['label'].tolist()

        self.transform = transform
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
        label = self.labels[index]
        label = torch.FloatTensor([label])

        if self.transform:
            array = self.transform(array)
            array = array.numpy()


        array = np.stack((array,)*3, axis=1)
        array = torch.FloatTensor(array)

        if label.item() == 1:
            weight = np.array([self.weights[1]])
            weight = torch.FloatTensor(weight)
        else:
            weight = np.array([self.weights[0]])
            weight = torch.FloatTensor(weight)

        return array, label, weight
