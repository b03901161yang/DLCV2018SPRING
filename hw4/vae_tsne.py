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

from sklearn.manifold import TSNE

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

    def forward_new(self, x):
        mu, var = self.encoder(x)
        code = self.reparameter(mu, var)
        #out = self.decoder(code)
        return code

class Face_for_test(Dataset):
    def __init__(self, dir, len_data):
        self.dir = dir
        self.len_data = len_data
    def __len__(self):
        return self.len_data
    def __getitem__(self, idx):
        img_idx = str(idx+40000).zfill(5)
        img_name = os.path.join(self.dir, img_idx + '.png')
        df = pd.read_csv('hw4_data/train.csv')
        label_list = df["Male"].tolist() #only choose male/female here
        label = label_list[idx]
        img = io.imread(img_name)
        img = img.astype(np.float32)
        img = img/255
        img_np = img
        label_list = np.array(label_list)
        # numpy: (H, W, C) -> torch: (C, H, W)
        img = img.transpose((2, 0, 1))
        img_tor = torch.from_numpy(img)
        return {'np': img_np, 'tor': img_tor, 'label':label}

def plot_tsne(net, dataloader, batch_size, guid, out_path):
    test_loss = 0
    num_of_batch = 0
    x_plot = []
    y_plot = []
    number_of_points = 500
    for batch, x in enumerate(dataloader):
        y = Variable(x['label'])
        x = Variable(x['tor'])

        if guid >= 0:   # move to GPU
            x = x.cuda(guid)
        if batch < number_of_points:
            x_encode = net.forward_new(x)
            #print('type encode x :', type(x_encode)) 
            #print('size encode x :', x_encode.size()) 
            #print('type y label : ',type(y))
            #print('size  y label :',  y label.size())
            x_encode = x_encode.data.cpu().numpy()
            y = y.data.cpu().numpy()
            x_encode = np.reshape(x_encode,(x_encode.shape[1],))

            x_plot.append(x_encode)
            y_plot.append(y)
        if batch == number_of_points:
            x_plot = np.array(x_plot)
            y_plot = np.array(y_plot)
            y_plot = np.reshape(y_plot,(y_plot.shape[0],))
            print('starting tsne')
            #print('x_plot x:', x_plot.shape)
            #print('y_plot:', y_plot.shape)
            vis_data = TSNE(n_components=2).fit_transform(x_plot)
            print('vis data shape',vis_data.shape)
            print('tsne finished')
            vis_data_x = vis_data[:,0]
            vis_data_y = vis_data[:,1]
            print('vis_data_x shape',vis_data_x.shape)
            cm = plt.cm.get_cmap('RdYlBu')
            sc = plt.scatter(vis_data_x, vis_data_y, c= y_plot, cmap = cm)
            plt.colorbar(sc)
            plt.savefig(os.path.join(out_path,'fig1_5.jpg'))
            #plt.show()
            print('1_5, image saved')
            break
    return test_loss

def main():
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    batch_size = 1
    torch.manual_seed(1)
    net = torch.load('vae_epoch_9.pt')

    net = net.cuda(0)

    dataset = Face_for_test(dir=os.path.join(in_path,'test'), len_data=2621)
    
    # create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    plot_tsne(net, dataloader, batch_size, 0, out_path)

if __name__ == "__main__":
  main()

