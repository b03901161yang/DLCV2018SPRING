from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import os
import pandas as pd 
from skimage import io, transform
import numpy as np 
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from typing import Tuple
import sys


from torchvision.utils import save_image
import torchvision.utils as vutils
myseed = 2343
np.random.seed(myseed)
torch.manual_seed(myseed)
torch.cuda.manual_seed_all(myseed)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.num_z = 128
        
        self.conv1 = nn.Sequential(nn.Conv2d(3, 128, 3, 1, 1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU())
        self.maxpooling1 = nn.MaxPool2d(2, padding=0)
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 5, 1, 2), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, 5, 1, 2), nn.ReLU())
        self.maxpooling2 = nn.MaxPool2d(2, padding=0)
        self.conv5 = nn.Sequential(nn.Conv2d(256, 512, 5, 1, 2), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(512, 512, 5, 1, 2), nn.ReLU())
        self.maxpooling3 = nn.MaxPool2d(2, padding=0)

        self.fc_encode1 = nn.Linear(8*8*512, self.num_z)
        self.fc_encode2 = nn.Linear(8*8*512, self.num_z)
        
        
        self.fc_decode = nn.Linear(self.num_z, 8*8*512)
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(512, 512, 4, 2, 1), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(512, 512, 4, 2, 1), nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(512,   3, 4, 2, 1), nn.Sigmoid())

    def encoder(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.maxpooling1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.maxpooling2(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.maxpooling3(out)
        #print(out)
        return self.fc_encode1(out.view(out.size(0), -1)), self.fc_encode2(out.view(out.size(0), -1))

    def decoder(self, x):
        out = self.fc_decode(x)
        out = self.deconv1(out.view(x.size(0), 512, 8, 8))
        out = self.deconv2(out)
        out = self.deconv3(out)
        return out
    
    def reparameter(self, mu, var):
        #print('Reparameter:', mu.shape)
        if self.training:
            e = Variable( torch.from_numpy(
                    np.random.normal(0, 1, (mu.shape[0], mu.shape[1]))).float())
            e = e.cuda(0)
            z = mu + var*e
        else:
            z = mu
        return z

    def forward(self, x):
        mu, var = self.encoder(x)
        code = self.reparameter(mu, var)
        out = self.decoder(code)
        return out, mu, var

in_path = sys.argv[1]
out_path = sys.argv[2]

net = torch.load('vae_epoch_9.pt')
net = net.cuda(0)

h = 8
w = 4
nz = 128

fixed_noise = Variable(torch.randn(h*w, nz))


fixed_noise = fixed_noise.cuda(0)
fake = net.decoder(fixed_noise)

#print('fake.data size :', fake.data.size())

vutils.save_image(fake.data, os.path.join(out_path,'fig1_4.jpg') ,nrow=8)

print('1_4, image saved')
