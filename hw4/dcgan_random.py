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
from tensorboardX import SummaryWriter 
import torchvision.utils as vutils

myseed = 2343
np.random.seed(myseed)
torch.manual_seed(myseed)
torch.cuda.manual_seed_all(myseed)


# code modified from : https://github.com/pytorch/examples/blob/master/dcgan/main.py

class generator(nn.Module): #similar to decoder of VAE
    def __init__(self):
        super(generator, self).__init__()
        self.guid = 0
        self.num_z = 128
        #self.fc_gen = nn.Linear(self.num_z, 8*8*512)
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(self.num_z, 512, 4, 1, 0,  bias=False), nn.BatchNorm2d(512) , nn.LeakyReLU(0.2, inplace=True))
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1,  bias=False), nn.BatchNorm2d(256) , nn.LeakyReLU(0.2, inplace=True))
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1,  bias=False), nn.BatchNorm2d(128) , nn.LeakyReLU(0.2, inplace=True))
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(128,  64, 4, 2, 1,  bias=False), nn.BatchNorm2d(64) , nn.LeakyReLU(0.2, inplace=True))
        self.deconv5 = nn.Sequential(nn.ConvTranspose2d(64 ,   3, 4, 2, 1,  bias=False), nn.Tanh()) #use tanh for generator
         
    def forward(self, x):
        #print('generator input size: ', x.size())
        out = x.view(x.size(0), 128, 1, 1)
        #print('input size after view: ', out.size())
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        out = self.deconv5(out)
        #print('generater output size: ',out.size())
        return out

class discriminator(nn.Module): #similar to decoder of VAE
    def __init__(self):
        super(discriminator, self).__init__()
        self.guid = 0
        self.num_z = 128
        self.conv1 = nn.Sequential(nn.Conv2d(3,    64, 4, 2, 1,    bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True)) #nn.BatchNorm2d?
        self.conv2 = nn.Sequential(nn.Conv2d(64,  128, 4, 2, 1,  bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1,  bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, 4, 4, 1,  bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(512,   1, 4, 4, 1,  bias=False), nn.Sigmoid())
    
    def forward(self, x):
        out = self.conv1(x)
        #print('conv1 out size: ',out.size())
        out = self.conv2(out)
        out = self.conv3(out)
        #print('conv3 out size: ',out.size())
        out = self.conv4(out)
        out = self.conv5(out)
        out = out.view(-1, 1).squeeze(1)
        #print('discriminator output size', out.size())
        return out

in_path = sys.argv[1]
out_path = sys.argv[2]


netG = generator()
netD = discriminator()

netG = torch.load('dcgan_gen_epoch_10.pt')
netD = torch.load('dcgan_dis_epoch_10.pt')
netG.cuda()
netD.cuda()
#print(netG)
#print(netD)

h = 8 #number of images in row
w = 4 #number of images in col
nz = 128

fixed_noise = torch.FloatTensor(h*w, nz, 1, 1).normal_(0, 1)
fixed_noise = fixed_noise.cuda()
fixed_noise = Variable(fixed_noise)

fixed_noise_stack = []
for i in range(4):
    fixed_noise_ = np.random.normal(0, 1, (h, nz))
    fixed_noise_stack.append(fixed_noise_)

fixed_noise_ = np.vstack((fixed_noise_stack[0] ,fixed_noise_stack[1],fixed_noise_stack[2],fixed_noise_stack[3])) # concatenation

#print('fixed_noise_ shape :', fixed_noise_.shape)

fixed_noise_ = np.reshape(fixed_noise_,(h*w, nz, 1, 1))
fixed_noise_ = (torch.from_numpy(fixed_noise_))
fixed_noise.data.copy_(fixed_noise_)

fake = netG(fixed_noise)
#print('fake.data size :', fake.data.size())

vutils.save_image(fake.data, os.path.join(out_path,'fig2_3.jpg') ,nrow=8)

print('2_3, image saved')
