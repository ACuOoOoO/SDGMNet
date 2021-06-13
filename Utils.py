'''
All utils are from https://github.com/DagnyT/hardnet
'''

import torch
import torch.nn.init
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import PIL
# resize image to size 32x32
def np_reshape(x):
    x_ = np.reshape(x, (32, 32, 1))
    return x_

def np_reshape64(x):
    x_ = np.reshape(x, (64, 64, 1))
    return x_

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps).detach()
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x

class L1Norm(nn.Module):
    def __init__(self):
        super(L1Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sum(torch.abs(x), dim = 1) + self.eps
        x= x / norm.expand_as(x)
        return x


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

class nn_transform_train(nn.Module):
    def __init__(self):
        super(nn_transform_train, self).__init__()
        self.transform = torch.nn.Sequential(transforms.RandomRotation(5,resample = PIL.Image.BILINEAR),
                                          transforms.RandomResizedCrop(32, scale = (0.9, 1.0),ratio = (0.9, 1.1)),
                                          transforms.Resize(32))
        
    def forward(self, input):
        x = input.to(torch.float)/255
        return self.transform(x)


class nn_transform_test(nn.Module):
    def __init__(self):
        super(nn_transform_test, self).__init__()
        self.transform = torch.nn.Sequential([transforms.Resize(32)])
        
    def forward(self, input):
        return self.transform(input)

transform_train = transforms.Compose([
    transforms.Lambda(np_reshape64),
    transforms.ToPILImage(),
    transforms.RandomRotation(5, resample=PIL.Image.BILINEAR, fill=(0,)),
    transforms.RandomResizedCrop(32, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
    transforms.Resize(32),
    transforms.ToTensor()])


transform_test = transforms.Compose([
    transforms.Lambda(np_reshape64),
    transforms.ToPILImage(),
    transforms.Resize(32),
    transforms.ToTensor()])

import os

class FileLogger:
    "Log text in file."
    def __init__(self, path):
        self.path = path
        if os.path.exists(path):
            os.makedirs(path)

    def log_string(self, file_name, string):
        """Stores log string in specified file."""
        text_file = open(self.path+file_name+".log", "a")
        text_file.write(string)
        text_file.close()

    def log_stats(self, file_name, text_to_save, value):
        """Stores log in specified file."""
        text_file = open(self.path+file_name+".log", "a")
        if type(value) == type(0):
            text_file.write(text_to_save+' '+str(value))
        else:
            text_file.write(text_to_save+' {:.5f}'.format(value))
        text_file.close()
