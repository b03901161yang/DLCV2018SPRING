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

from vae_train import *
from torchvision.utils import save_image
import torchvision.utils as vutils
class Face_for_test(Dataset):
    def __init__(self, dir, len_data):
        self.dir = dir
        self.len_data = len_data
    def __len__(self):
        return self.len_data
    def __getitem__(self, idx):
        img_idx = str(idx+40000).zfill(5)
        img_name = os.path.join(self.dir, img_idx + '.png')
        img = io.imread(img_name)
        img = img.astype(np.float32)
        img = img/255
        img_np = img
        
        # numpy: (H, W, C) -> torch: (C, H, W)
        img = img.transpose((2, 0, 1))
        img_tor = torch.from_numpy(img)
        return {'np': img_np, 'tor': img_tor}

def predict(net, dataloader, batch_size, guid, out_path):
    test_loss = 0
    image_out = torch.FloatTensor(20, 3, 64, 64)
    image_out = image_out.cuda()
    num_of_batch = 0
    mu_all = 0
    for batch, x in enumerate(dataloader):
        x = Variable(x['tor'])

        if guid >= 0:   # move to GPU
            x = x.cuda(guid)

        recon_x, mu, logvar = net(x)
        #print('mu size:', mu.size()) 
        #print('mu: ', mu[0].data.tolist()[0])
        mu_all += mu[0].data.tolist()[0]

        mse = F.mse_loss(recon_x, x)
        #print(mse.data[0])
        
        #test_loss += mse #QQ
        test_loss += mse.data.tolist()[0]
        #print('recon_x: ',recon_x.size())
        if batch < 10: #pick 10 
            image_out[batch,:,:,:] = x.data[0,:,:,:]
            image_out[batch+10,:,:,:] = recon_x.data[0,:,:,:]
        num_of_batch = batch
    print('nubmer of testing image', len(dataloader.dataset))
    print('type of image out: ',type(image_out))
    vutils.save_image(image_out, os.path.join(out_path,'fig1_3.jpg') ,nrow=10)
    print('1_3 ,image saved')
    #print(num_of_batch)
    test_loss /= num_of_batch
    print('====> Test set loss: {:.4f}'.format(test_loss))

    mu_all /= num_of_batch
    print('====> mu all : {:.4f}'.format(mu_all))
    return test_loss

def main():
    guid = 0
    myseed = 1
    batch_size = 32
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    torch.manual_seed(myseed)
    net = torch.load('vae_epoch_9.pt')

    if guid >= 0:
        net = net.cuda(guid)

    dataset = Face_for_test(dir=os.path.join(in_path,'test'), len_data=2621)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    predict(net, dataloader, batch_size, guid, out_path)

if __name__ == "__main__":
  main()

