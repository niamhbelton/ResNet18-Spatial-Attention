import shutil
import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import model as model
from dataloader import Dataset
import torch.nn.functional as F
import pandas as pd
from sklearn import metrics


def evaluate(args):

    test_dataset = Dataset(args.data_directory, args.task, args.plane, test = True, transform = None)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=-True, num_workers=11, drop_last=False)

    model = torch.load(args.model_directory + args.model_name)

    _ = model.eval()

    if torch.cuda.is_available():
        model.cuda()

    y_trues = []
    y_preds = []
    losses = []
    log_every=20
    for i, (image, label, weight) in enumerate(test_loader):

        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
            weight = weight.cuda()

        label = label[0]
        weight = weight[0]

        prediction = model.forward(image.float()).squeeze(0)

        loss = torch.nn.BCEWithLogitsLoss(weight=weight)(prediction, label)

        loss_value = loss.item()
        losses.append(loss_value)

        probas = torch.sigmoid(prediction)

        y_trues.append(int(label[0]))
        y_preds.append(probas[0].item())

        try:
            auc = metrics.roc_auc_score(y_trues, y_preds)
        except:
            auc = 0.5


        if (i % log_every == 0) & (i > 0):
            print('''[Single batch number : {0} / {1} ] | avg test loss {2} | test auc : {3} '''.
                  format(
                      i,
                      len(test_loader),
                      np.round(np.mean(losses), 4),
                      np.round(auc, 4),
                  )
                  )


    test_loss = np.round(np.mean(losses), 4)
    test_auc = np.round(auc, 4)
    print('Test loss is {}'.format(test_loss))
    print('Test AUC is {}'.format(test_auc))

    return test_loss, test_auc



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, required=True,
                        choices=['abnormal', 'acl', 'meniscus'])
    parser.add_argument('-p', '--plane', type=str, required = True, default=None)
    parser.add_argument('--model_name', type=str, required = True, default=None)
    parser.add_argument('-md', '--model_directory', type=str, default='/models/')
    parser.add_argument('-d', '--data_directory', type=str, default='/home/Documents/data/')
    parser.add_argument('--log_every', type=int, default=100)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    evaluate(args)
