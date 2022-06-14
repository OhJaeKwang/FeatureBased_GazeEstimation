import torch
from torch import nn
from torch.nn import Module
from torchsummary import summary

import numpy as np
import cv2
import json
import math
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class GazeEstimator(Module):   # pretrain model --> mpii_ver2
    def __init__(self):
        super(GazeEstimator, self).__init__()

        self.fc1 = nn.Linear(in_features=100, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=3)

        self.actv = nn.LeakyReLU(inplace=True)
        

    def forward(self, x):
        
        x = self.fc1(x)
        x = self.actv(x)
        x = self.fc2(x)
        x = self.actv(x)
        x = self.fc3(x)

        return x

# 바로전 
# class GazeEstimator(Module):   # pretrain model --> mpii_ver2
#     def __init__(self):
#         super(GazeEstimator, self).__init__()

#         self.fc1 = nn.Linear(in_features=100, out_features=64)
#         self.fc2 = nn.Linear(in_features=64, out_features=3)
        
        
#         self.actv = nn.LeakyReLU(inplace=True)
        

#     def forward(self, x):
        
#         x = self.fc1(x)
#         x = self.actv(x)
#         x = self.fc2(x)

#         return x
