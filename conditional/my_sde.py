import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import abc
import math, tqdm, time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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

    def loss_fn(self, x_0, label, alpha=0.):
        eps = 1e-5
        x_0 = x_0.squeeze(1) 
        t = torch.rand(x_0.shape[0], device=device) * (1. - eps) + eps
        z = torch.randn_like(x_0)

        ave, std = self.marginal_prob(t)
        x_t = x_0 * ave + z * std
        output, loss_bt1 = self.model(x_t, t, label, masked=True, weight_loss=True)
        loss = (output * std + z).square().mean() + alpha * loss_bt1
        return loss

    def train(self, dataloader, n_epochs, lr=1e-3, ema=0.9, alpha=1e-7, lambda_=1e-5):
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=lr,
                                     weight_decay=lambda_)
        ema = EMA(ema)
        ema.register(self.model)
        epoch = tqdm.trange(n_epochs)
        loss_ls = []
        for ep in epoch:
            for idx, (batch_x, label) in enumerate(dataloader):
                loss_batch = self.loss_fn(batch_x, label, alpha=alpha)
                loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                optimizer.step()
                ema.update(self.model)
                optimizer.zero_grad()
                loss_ls += loss_batch.item(),
                epoch.set_description(f"Loss: {loss_batch.item():.3g}")
            if ep%500 == 0:
                torch.save(self.model.state_dict(), f'score_model/model/tmp/model_{ep}.pth')
        return loss_ls

    def r_sde(self, x, t, w, c, probability_flow=False):
        drift, diffusion = self.sde(x, t)
        if x.dim() == 3:
            x = x.squeeze(0)
        c_0 = torch.tensor([0], device=device).repeat(x.shape[0], 1)
        score = (1+w) * self.model(x, t, c=c, masked=False) - w * self.model(x, t, c=c_0, masked=False)
        drift = drift - diffusion[:, None] ** 2 * score * (0.5 if probability_flow else 1.)
        diffusion = 0. if probability_flow else diffusion
        return drift, diffusion

    @torch.no_grad()
    def sample(self, n_samples, n_steps, 
               delta_t=None, x_T=None, dim=2, type='euler',
               w=0, c=0):
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
                f_, g_ = self.r_sde(x_t, t, w, c)
                mean_x = x_t - f_ * delta_t 
                x_t = mean_x + math.sqrt(delta_t) * g_ * torch.randn_like(x_t)
            return mean_x
        
        elif type == 'euler_ode':
            for n in range(n_steps):
                t = time_step[n].unsqueeze(-1)
                f_, g_ = self.r_sde(x_t, t, w, c, probability_flow=True)
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



class SDE_nc(abc.ABC):
    def __init__(self, score_model) -> None:
        super().__init__()
        self.model = score_model

    @abc.abstractclassmethod
    def sde(self, x, t):
        pass

    @abc.abstractclassmethod
    def marginal_prob(self, t):
        pass

    def loss_fn(self, x_0, noise_consistent=False):
        eps = 1e-5
        t = torch.rand(x_0.shape[0], device=device) * (1. - eps) + eps
        z = torch.randn_like(x_0)

        ave, std = self.marginal_prob(t)
        x_t = x_0 * ave + z * std

        _, gt = self.sde(x_t, t)

        output = self.model(x_t, t)
        if noise_consistent:
            target = z * gt[:, None]
        else:
            target = z

        loss = (output * std + target).square().mean()
        return loss

    def train(self, dataloader, n_epochs, lr=1e-3, noise_consistent=False):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        ema = EMA(0.95)
        ema.register(self.model)
        epoch = tqdm.trange(n_epochs) 
        loss_ls = []
        for ep in epoch:
            for idx, batch_x in enumerate(dataloader):
                loss_batch = self.loss_fn(batch_x, noise_consistent)
                loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                optimizer.step()
                ema.update(self.model)
                optimizer.zero_grad()
                loss_ls += loss_batch.item(),
                epoch.set_description(f"Loss: {loss_batch.item():.3g}")
            if ep%500 == 0:
                torch.save(self.model.state_dict(), f'score_model/model/tmp/model_{ep}.pth')
        return loss_ls

    def r_sde(self, x, t, probability_flow=False, noise_consistent=False, 
              coef_net=1, coef_noise=1):
        
        drift, diffusion = self.sde(x, t)
        score = self.model(x, t)
        if noise_consistent:
            drift = drift - diffusion[:, None] ** 1 * score * (0.5 if probability_flow else 1.) * coef_net
        else:
            drift = drift - diffusion[:, None] ** 2 * score * (0.5 if probability_flow else 1.) * coef_net

        diffusion = 0. if probability_flow else diffusion
        return drift, diffusion*coef_noise

    @torch.no_grad()
    def sample(self, n_samples, n_steps, delta_t=None, x_T=None, dim=2, 
               type='Langevin', path_record=False, noise_consistent=False,
               coef_net=1, coef_noise=1):
        eps = 1e-4
        if delta_t is None:
            delta_t = (1. - eps)/ n_steps
        if x_T is None:
            x_T = torch.randn([n_samples, dim], device=device)
        
        time_step = torch.linspace(1., eps, n_steps, device=device)
        x_t = x_T

        if type == 'Langevin':
            for t in tqdm.tqdm(range(n_steps)):
                Score = self.model(x_t, time_step[t].unsqueeze(-1))
                x_t = x_t + 0.5 * delta_t * Score + math.sqrt(delta_t) * torch.randn_like(x_t)
            return x_t

        else:
            path_data = []
            path_data.append(x_t.cpu().numpy())
            for n in range(n_steps):
                t = time_step[n].unsqueeze(-1)
                if type == 'euler':
                    f_, g_ = self.r_sde(x_t, t, noise_consistent=noise_consistent, 
                                        coef_net=coef_net, coef_noise=coef_noise)
                elif type == 'euler_ode':
                    f_, g_ = self.r_sde(x_t, t, probability_flow=True, noise_consistent=noise_consistent, 
                                        coef_net=coef_net, coef_noise=coef_noise)
                mean_x = x_t - f_ * delta_t
                x_t = mean_x + math.sqrt(delta_t) * g_ * torch.randn_like(x_t)                  # Gussian
                path_data.append(x_t.cpu().numpy()[0,:])
            if path_record:
                return mean_x, path_data
            else:
                return mean_x
        
class VPSDE_nc(SDE_nc):
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