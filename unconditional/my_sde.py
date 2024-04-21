import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import abc
import math, tqdm, time

# device = torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

class TimeEncoding(nn.Module):
    def __init__(self, embed_dim, scale=1.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        if x.shape == torch.Size([]):
            x = x.unsqueeze(-1)
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class MLP(nn.Module):
    def __init__(self, n_node=32, amp=1):
        super().__init__()
        self.n_node = n_node
        self.amp = amp

        self.embed = TimeEncoding(n_node)
        self.linears = nn.ModuleList(
            [
                nn.Linear(2, self.n_node),
                nn.ReLU(),
                nn.Linear(self.n_node, self.n_node),
                nn.ReLU(), 
                nn.Linear(self.n_node, 2),
            ]
        )
    def forward(self, x, t):
        t_embedding = self.embed(t)
        for idx in range(2):
            x = self.linears[idx * 2](x)
            x += t_embedding
            x = self.linears[idx * 2 + 1](x)
        x = self.linears[-1](x)
        return x * self.amp


class MLP_noise(nn.Module):
    def __init__(self, n_node=32, n_std=0.1, n_lim=0.2, amp=10):
        super().__init__()
        self.n_node = n_node
        self.std = n_std
        self.lim = n_lim
        self.amp = amp

        self.embed = TimeEncoding(n_node)
        self.linears = nn.ModuleList(
            [
                nn.Linear(2, self.n_node),
                nn.ReLU(),
                nn.Linear(self.n_node, self.n_node),
                nn.ReLU(), 
                nn.Linear(self.n_node, 2),
            ]
        )
    def forward(self, x, t, threshold=1.5):
        t_embedding = self.embed(t)
        
        loss_bt1 = 0
        for idx in range(2):
            x = x @ add_noise(self.linears[idx * 2].weight.T, n_std=self.std, n_lim=self.lim) + \
                    add_noise(self.linears[idx * 2].bias, n_std=self.std, n_lim=self.lim)
            
            x += t_embedding
            x = self.linears[idx * 2 + 1](x)

            loss_bt1 += torch.sum(x[x>threshold])

        x = x @ add_noise(self.linears[-1].weight.T, n_std=self.std, n_lim=self.lim) + \
                add_noise(self.linears[-1].bias, n_std=self.std, n_lim=self.lim)
        return x * self.amp, loss_bt1

def add_noise(data, n_std=0.1, n_lim=0.2):
    noise = torch.randn_like(data).to(device) * n_std
    mask = noise.abs() < n_lim
    noise_mask = (torch.rand_like(data).to(device) *2 -1) * n_lim
    noise_mask[mask] = noise[mask]

    return data * (1 + noise_mask)

def calDistance(dataset, points, bins_p=100):
    ori_points = dataset.cpu().numpy().squeeze()
    ge_points = points.cpu().numpy().squeeze()

    mat_l = []
    bins_p = bins_p
    for plot_points in [ori_points, ge_points]:
        mat_p = np.histogram2d(plot_points[:,0], plot_points[:,1], 
                               bins=bins_p,
                               range=[[-1, 1], [-1, 1]],)
        mat_l.append(mat_p[0]/plot_points.shape[0])

    return ((mat_l[0]-mat_l[1])**2).sum()**0.5

    
class SDE(abc.ABC):
    def __init__(self, score_model) -> None:
        super().__init__()
        self.model = score_model

    @abc.abstractclassmethod
    def sde(self, x, t):
        pass

    @abc.abstractclassmethod
    def marginal_prob(self, t):
        pass

    def loss_fn(self, x_0, alpha=1):
        eps = 1e-5
        x_0 = x_0.squeeze(1) 
        t = torch.rand(x_0.shape[0], device=device) * (1. - eps) + eps
        z = torch.randn_like(x_0)

        ave, std = self.marginal_prob(t)
        x_t = x_0 * ave + z * std
        output, loss_bt1 = self.model(x_t, t)
        loss = (output * std + z).square().mean() + alpha * loss_bt1
        return loss

    def train(self, dataloader, n_epochs, lr=1e-3, alpha=1, lambda_=1e-4):
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=lr,
                                     weight_decay=lambda_)
        ema = EMA(0.95)
        ema.register(self.model)
        epoch = tqdm.trange(n_epochs)
        loss_ls = []
        for ep in epoch:
            for idx, batch_x in enumerate(dataloader):
                loss_batch = self.loss_fn(batch_x, alpha=alpha)
                loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                optimizer.step()
                ema.update(self.model)
                optimizer.zero_grad()
                loss_ls += loss_batch.item(),
                epoch.set_description(f"Loss: {loss_batch.item():.3g}")
            # if ep%500 == 0:
            #     torch.save(self.model.state_dict(), f'score_model/model/tmp/model_{ep}.pth')
        return loss_ls

    def r_sde(self, x, t, probability_flow=False):
        drift, diffusion = self.sde(x, t)
        score = self.model(x, t)
        drift = drift - diffusion[:, None] ** 2 * score * (0.5 if probability_flow else 1.)
        diffusion = 0. if probability_flow else diffusion
        return drift, diffusion

    @torch.no_grad()
    def sample(self, n_samples, n_steps, delta_t=None, x_T=None, dim=2, type='euler_ode'):
        eps = 1e-4
        if delta_t is None:
            delta_t = (1. - eps)/ n_steps
        if x_T is None:
            x_T = torch.randn([n_samples, dim], device=device)
        
        time_step = torch.linspace(1., eps, n_steps, device=device)
        x_t = x_T

        if type == 'euler':
            for n in range(n_steps):
                t = time_step[n].unsqueeze(-1)
                f_, g_ = self.r_sde(x_t, t)
                mean_x = x_t - f_ * delta_t
                x_t = mean_x + math.sqrt(delta_t) * g_ * torch.randn_like(x_t)                  # Gussian
            return mean_x
        
        elif type == 'euler_ode':
            for n in range(n_steps):
                t = time_step[n].unsqueeze(-1)
                f_, g_ = self.r_sde(x_t, t, probability_flow=True)
                mean_x = x_t - f_ * delta_t
                x_t = mean_x + math.sqrt(delta_t) * g_ * torch.randn_like(x_t)
            return mean_x


class VESDE(SDE):
    def __init__(self, score_model, sigma=10) -> None:
        super().__init__(score_model)
        self.sigma = sigma

    def sde(self, x, t):
        drift = 0.
        diffusion = self.sigma ** t
        return drift, diffusion

    def marginal_prob(self, t):
        mean = torch.ones_like(t)
        std = torch.sqrt((self.sigma**(2 * t) - 1.) / 2. / np.log(self.sigma))
        return mean[:, None], std[:, None]

class VPSDE(SDE):
    def __init__(self, score_model, beta_min=0.001, beta_max=1) -> None:
        super().__init__(score_model)
        self.beta_0 = beta_min
        self.beta_1 = beta_max

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        if beta_t.shape == torch.Size([]):
            beta_t = beta_t.unsqueeze(-1)
        drift = -0.5 * beta_t[:, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion[:, None]

    def marginal_prob(self, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None])
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std[:, None]


class EMA(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict



