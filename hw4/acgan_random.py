from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import os
import pandas as pd 
from skimage import io
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import sys
import numpy as np
import acgan_model
import torchvision.utils as vutils
#code modified from : https://github.com/kimhc6028/acgan-pytorch
# code modified from https://github.com/pytorch/examples/blob/master/dcgan/main.py
myseed = 2343
np.random.seed(myseed)
torch.manual_seed(myseed)
torch.cuda.manual_seed_all(myseed)


pairNum = 10
nz = 100
nb_label = 2

ngf = 64
ndf = 64
nc = 3

in_path = sys.argv[1]
out_path = sys.argv[2]

netG = torch.load('netG_epoch_22.pt')
netD = torch.load('netD_epoch_22.pt')
netG.cuda()
netD.cuda()
#print(netG)
#print(netD)


fixed_noise = torch.FloatTensor(2*pairNum, nz, 1, 1).normal_(0, 1)

real_label = 1
fake_label = 0

fixed_noise = fixed_noise.cuda()
fixed_noise = Variable(fixed_noise)

fixed_noise_ = np.random.normal(0, 1, (pairNum, nz))
fixed_noise_ = np.vstack((fixed_noise_,fixed_noise_)) # concatenation two fixed_noise
label_onehot = np.zeros((2*pairNum, nb_label))
label_onehot[np.arange(pairNum), 0] = 1
label_onehot[np.arange(pairNum,2*pairNum), 1] = 1
fixed_noise_[np.arange(2*pairNum), :nb_label] = label_onehot[np.arange(2*pairNum)]

fixed_noise_ = np.reshape(fixed_noise_,(2*pairNum, nz, 1, 1))
fixed_noise_ = (torch.from_numpy(fixed_noise_))
fixed_noise.data.copy_(fixed_noise_)

fake = netG(fixed_noise)
vutils.save_image(fake.data, os.path.join(out_path,'fig3_3.jpg') ,nrow=10)

print('3_3, image saved')
