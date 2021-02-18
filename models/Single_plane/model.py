import torch
import torch.nn as nn
from torchvision import models
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_model = models.resnet18(pretrained=True)
        self.conv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.soft = nn.Softmax(2)
        self.classifer = nn.Linear(1000, 1)
        
    def tile(a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        if torch.cuda.is_available():
            a = a.cuda()
          #  dim = torch.Tensor(dim).cuda()
            order_index = order_index.cuda()
        return torch.index_select(a, dim, order_index)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        x = self.pretrained_model.conv1(x)
        x = self.pretrained_model.bn1(x)
        x = self.pretrained_model.maxpool(x)
        x = self.pretrained_model.layer1(x)
        x = self.pretrained_model.layer2(x)
        x = self.pretrained_model.layer3(x)
        x = self.pretrained_model.layer4(x)
        a = self.conv(x)
        a =  self.soft(a.view(*a.size()[:2], -1)).view_as(a)
        m = torch.max(a.flatten(2), 2).values
        b= Net.tile(m, 1, 64)
        c = a.flatten(2).flatten(1) / b
        d = torch.reshape(c, (a.shape[0],512,8,8))
        o = x*d 
        out= self.pretrained_model.avgpool(o)
        out = out.squeeze()
        out = self.pretrained_model.fc(out)
        output = torch.max(out, 0, keepdim=True)[0]
        output = self.classifer(output)

        return output
