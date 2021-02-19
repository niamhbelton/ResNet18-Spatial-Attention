
import shutil
import os
import time
from datetime import datetime
import argparse
import numpy as np
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import model
from torchsample.transforms import RandomRotate, RandomTranslate, RandomFlip, ToTensor, Compose, RandomAffine
from torchvision import transforms
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import math
from dataloader import Dataset
import pandas as pd
from sklearn import metrics


def train_model(model, train_loader, epoch, num_epochs, optimizer, writer, current_lr, log_every=100):
    _ = model.train()

    if torch.cuda.is_available():
        model.cuda()

    y_preds = []
    y_trues = []
    losses = []
    for i, (image, label, weight) in enumerate(train_loader):
        optimizer.zero_grad()

        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
            weight = weight.cuda()

        label = label[0]
        weight = weight[0]


        prediction = model.forward(image.float()).squeeze(0)


        loss = torch.nn.BCEWithLogitsLoss(weight=weight)(prediction, label)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)

        probas = torch.sigmoid(prediction)

        y_trues.append(int(label[0]))
        y_preds.append(probas[0].item())

        try:
            auc = metrics.roc_auc_score(y_trues, y_preds)
        except:
            auc = 0.5

        writer.add_scalar('Train/Loss', loss_value,
                          epoch * len(train_loader) + i)
        writer.add_scalar('Train/AUC', auc, epoch * len(train_loader) + i)

        if (i % log_every == 0) & (i > 0):
            print('''[Epoch: {0} / {1} |Single batch number : {2} / {3} ]| avg train loss {4} | train auc : {5} | lr : {6}'''.
                  format(
                      epoch + 1,
                      num_epochs,
                      i,
                      len(train_loader),
                      np.round(np.mean(losses), 4),
                      np.round(auc, 4),
                      current_lr
                  )
                  )

    writer.add_scalar('Train/AUC_epoch', auc, epoch + i)

    train_loss_epoch = np.round(np.mean(losses), 4)
    train_auc_epoch = np.round(auc, 4)
    return train_loss_epoch, train_auc_epoch


def evaluate_model(model, val_loader, epoch, num_epochs, writer, current_lr, log_every=20):
    _ = model.eval()

    if torch.cuda.is_available():
        model.cuda()

    y_trues = []
    y_preds = []
    losses = []
    for i, (image, label, weight) in enumerate(val_loader):

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

        writer.add_scalar('Val/Loss', loss_value, epoch * len(val_loader) + i)
        writer.add_scalar('Val/AUC', auc, epoch * len(val_loader) + i)

        if (i % log_every == 0) & (i > 0):
            print('''[Epoch: {0} / {1} |Single batch number : {2} / {3} ] | avg val loss {4} | val auc : {5} | lr : {6}'''.
                  format(
                      epoch + 1,
                      num_epochs,
                      i,
                      len(val_loader),
                      np.round(np.mean(losses), 4),
                      np.round(auc, 4),
                      current_lr
                  )
                  )

    writer.add_scalar('Val/AUC_epoch', auc, epoch + i)

    val_loss_epoch = np.round(np.mean(losses), 4)
    val_auc_epoch = np.round(auc, 4)

    return val_loss_epoch, val_auc_epoch

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def run(args):
    indexes = list(range(0,1130))
    random.seed(26)
    random.shuffle(indexes)
    ind = math.floor(len(indexes) / 8)
    for fold in range(0,8):
        valid_ind = indexes[ind*(fold):ind*(fold+1)]
        train_ind = np.setdiff1d(indexes,valid_ind)


        log_root_folder = "./logs/{0}/".format(args.task)


        now = datetime.now()
        logdir = log_root_folder + now.strftime("%Y%m%d-%H%M%S") + "/"
        os.makedirs(logdir)
        try:
            os.makedirs('./models/')
        except:
            pass

        writer = SummaryWriter(logdir)

        augmentor = Compose([
            transforms.Lambda(lambda x: torch.Tensor(x)),
            RandomRotate(25),
            RandomTranslate([0.11, 0.11]),
            RandomFlip(),
        ])


        net = model.Net()


        if torch.cuda.is_available():
            net = net.cuda()

        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.1)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=4, factor=.3, threshold=1e-4, verbose=True)

        best_val_loss = float('inf')
        best_val_auc = float(0)
        num_epochs = args.epochs
        iteration_change_loss = 0
        patience = args.patience
        log_every = args.log_every

        t_start_training = time.time()
        train_dataset = Dataset(args.directory, args.task, args.plane,
                         test= False, transform=augmentor, indexes = train_ind)
        valid_dataset = Dataset(
            args.directory, args.task, args.plane, test =False, transform = None, indexes = valid_ind)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=True, num_workers=11, drop_last=False)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=1, shuffle=-True, num_workers=11, drop_last=False)



        for epoch in range(num_epochs):
            current_lr = get_lr(optimizer)

            t_start = time.time()

            train_loss, train_auc = train_model(
                net, train_loader, epoch, num_epochs, optimizer, writer, current_lr, log_every)
            val_loss, val_auc = evaluate_model(
                net, valid_loader, epoch, num_epochs, writer, current_lr)

            scheduler.step(val_loss)

            t_end = time.time()
            delta = t_end - t_start

            print("fold : {0} | train loss : {1} | train auc {2} | val loss {3} | val auc {4} | elapsed time {5} s".format(
                fold, train_loss, train_auc, val_loss, val_auc, delta))

            iteration_change_loss += 1
            print('-' * 30)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                file_name = f'model_fold{fold}_{args.prefix_name}_{args.task}_val_auc_{val_auc:0.4f}_train_auc_{train_auc:0.4f}_epoch_{epoch+1}.pth'
                for f in os.listdir('./models/'):
                    if (args.task in f) and (args.prefix_name in f) and ('fold'+str(fold) in f) :
                        os.remove(f'./models/{f}')
                torch.save(net, f'./models/{file_name}')



            if val_loss < best_val_loss:
                best_val_loss = val_loss
                iteration_change_loss = 0

            if iteration_change_loss == patience:
                print('Early stopping after {0} iterations without the decrease of the val loss'.
                      format(iteration_change_loss))
                break


    t_end_training = time.time()
    print(f'training took {t_end_training - t_start_training} s')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, required=True,
                        choices=['abnormal', 'acl', 'meniscus'])
    parser.add_argument('--prefix_name', type=str, required=True)
    parser.add_argument('-p', '--plane', type=str, required = True, default=None)
    parser.add_argument('-d', '--directory', type=str, required = False, default='/content/data/')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    if args.seed != None:
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark = False
    run(args)
