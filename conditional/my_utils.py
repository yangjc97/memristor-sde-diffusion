import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from torch.utils.data import Dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class PixelSampler():
    def __init__(self, path, threshold=100):
        image = Image.open(path).convert("I")                # load, convert
        image = np.array(image) < threshold                  # screen pixels
        points = np.stack(np.where(image), 1)                # location of points
        points_scaled = (points / image.shape - 0.5) * 2     # scale to [-1, 1]

        self.image = image
        self.points = points
        self.points_scaled = points_scaled

    @property
    def size(self):
        return np.array(self.image.shape)

    def plot(self, points=None):
        if points is None:
            plt.imshow(self.image, cmap="gray")
        else:
            slot = np.zeros(self.size)
            slot[points[:, 0], points[:, 1]] = 1
            plt.imshow(slot, cmap="gray")

    def sample(self, n_samples=10):
        choices = np.arange(len(self.points))
        choices = np.random.choice(choices, n_samples)
        points_sample = self.points_scaled[choices]
        # Change direction
        points_sample[:, [0,1]] = points_sample[:, [1,0]]
        points_sample[:, 1] = -points_sample[:, 1]
        return points_sample



class TimeEncoding(nn.Module):
    def __init__(self, embed_dim, scale=1.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        if x.shape == torch.Size([]):
            x = x.unsqueeze(-1)
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)
    
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def __len__(self):
        return len(self.data)
    
    def get_data(self):
        return self.data
    
    def get_labels(self):
        return self.labels
    

def getdataset(iter,numclass):
    cmd=''
    for i,j in enumerate(numclass):
        cmd +="("+"y=="+str(j)+")"
        if i!=(len(numclass)-1):
            cmd +="^"
    for i,(X,y)in enumerate(iter):
        if i==0 :
            index=np.where(eval(cmd))
            x_out=X[index]
            y_out=y[index]
        else:
            index=np.where(eval(cmd))
            x_out=torch.cat([x_out,X[index]],dim=0)
            y_out=torch.cat([y_out,y[index]],dim=0)
    for i,j in  enumerate(numclass):
        index=np.where(y_out==j)
        y_out[index]=i
    return x_out,y_out


def add_noise(data, n_std=0.1, n_lim=0.2):
    noise = torch.randn_like(data).to(device) * n_std
    mask = noise.abs() < n_lim
    noise_mask = (torch.rand_like(data).to(device) *2 -1) * n_lim   # large noise become uniform distribution
    noise_mask[mask] = noise[mask]

    return data * (1 + noise_mask)


import scipy

def cal_KL(points, gauss_data, bins=50, plot=True, ax=None):
    points_ = points.squeeze(0).detach().cpu().numpy().flatten()
    gauss_data_ = gauss_data.squeeze(0).detach().cpu().numpy().flatten()

    points_hist = np.histogram(points_, bins=bins, range=(-4,4), density=True)
    gauss_data_hist = np.histogram(gauss_data_, bins=bins, range=(-4,4), density=True)

    epsilon = 1e-10  # small constant to avoid division by zero
    px = points_hist[0] / np.sum(points_hist[0]) + epsilon
    py = gauss_data_hist[0] / np.sum(gauss_data_hist[0]) + epsilon

    KL = scipy.stats.entropy(px, py) 

    if plot:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
            ax.set_title(f'KL divergence: {KL:.4f}')
        width = points_hist[1][1] - points_hist[1][0]
        ax.bar(points_hist[1][:-1], px, width=width, alpha=0.5, label='generated')
        ax.bar(gauss_data_hist[1][:-1], py, width=width, alpha=0.5, label='samples')
        ax.legend()
    return KL