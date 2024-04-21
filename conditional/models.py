import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from my_utils import TimeEncoding, EmbedFC, add_noise



class MLP(nn.Module):
    def __init__(self, n_node=32):
        super().__init__()
        self.n_node = n_node
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
        return x



class MLP_diffusion(nn.Module):
    def __init__(self, n_dim=2, n_node=16, 
                 n_std=0, n_lim=0, amp=1, 
                 cond=False, n_classes=10,
                 bias=False):
        super().__init__()
        self.n_dim = n_dim
        self.n_node = n_node
        self.std = n_std
        self.lim = n_lim
        self.amp = amp
        self.bias_flag = bias

        self.cond = cond

        if self.cond:
            self.n_classes = n_classes
            self.conEmbed = EmbedFC(self.n_classes, self.n_node)

        self.embed = TimeEncoding(n_node)
        self.linears = nn.ModuleList(
            [
                nn.Linear(self.n_dim, self.n_node, bias=bias),
                nn.ReLU(),
                nn.Linear(self.n_node, self.n_node, bias=bias),
                nn.ReLU(), 
                nn.Linear(self.n_node, self.n_dim, bias=bias),
            ]
        )
    def forward(self, x, t, c=None, masked=False, threshold=1.5, weight_loss=False):
        t_embedding = self.embed(t)
        
        if self.cond:
            if masked:
                mask = torch.rand(c.shape, device=c.device)
                mask[mask >  0.1] = 1
                mask[mask <= 0.1] = 0
                c_onehot = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
                c_embedding = self.conEmbed(c_onehot) * mask.repeat(1, self.n_node)
            else:
                c_onehot = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
                c_embedding = self.conEmbed(c_onehot)
        else:
            c_embedding = torch.zeros_like(t_embedding, device=t_embedding.device)

        loss_bt1 = 0
        if self.bias_flag:
            x = x @ add_noise(self.linears[0].weight.T, n_std=self.std, n_lim=self.lim) + \
                add_noise(self.linears[0].bias, n_std=self.std, n_lim=self.lim)
        else:
            x = x @ add_noise(self.linears[0].weight.T, n_std=self.std, n_lim=self.lim)
        x += t_embedding + c_embedding
        x = self.linears[1](x)

        loss_bt1 += torch.sum(x[x>threshold])

        if self.bias_flag:
            x = x @ add_noise(self.linears[2].weight.T, n_std=self.std, n_lim=self.lim) + \
                add_noise(self.linears[2].bias, n_std=self.std, n_lim=self.lim)
        else:
            x = x @ add_noise(self.linears[2].weight.T, n_std=self.std, n_lim=self.lim)
        x += t_embedding + c_embedding
        x = self.linears[3](x)

        loss_bt1 += torch.sum(x[x>threshold])
        if self.bias_flag:
            x = x @ add_noise(self.linears[4].weight.T, n_std=self.std, n_lim=self.lim) + \
                add_noise(self.linears[4].bias, n_std=self.std, n_lim=self.lim)
        else:
            x = x @ add_noise(self.linears[4].weight.T, n_std=self.std, n_lim=self.lim)
        
        if weight_loss:
            return x * self.amp, loss_bt1
        else:
            return x * self.amp
        


class MLP_c(nn.Module):
    def __init__(self, n_node=32, amp=1):
        super().__init__()
        self.n_node = n_node
        self.n_classes = 3
        self.amp = amp

        self.conEmbed = EmbedFC(self.n_classes, self.n_node)
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
    def forward(self, x, t, c, masked=False, threshold=1.5, weight_loss=False):
        t_embedding = self.embed(t)
        
        if masked:
            mask = torch.rand(c.shape, device=c.device)
            mask[mask >  0.1] = 1
            mask[mask <= 0.1] = 0
            c_onehot = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
            c_embedding = self.conEmbed(c_onehot) * mask.repeat(1, self.n_node)
        else:
            c_onehot = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
            c_embedding = self.conEmbed(c_onehot)

        loss_bt1 = 0

        x = self.linears[0](x)
        x += t_embedding + c_embedding
        x = self.linears[1](x)

        loss_bt1 += torch.sum(x[x>threshold])

        x = self.linears[2](x)
        x += t_embedding + c_embedding
        x = self.linears[3](x)

        loss_bt1 += torch.sum(x[x>threshold])

        x = self.linears[4](x)
        if weight_loss:
            return x * self.amp, loss_bt1
        else:
            return x * self.amp

class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, num_labels=0, re_size=28):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, num_labels)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, num_labels)
        
        self.re_size = re_size

    def forward(self, x, c=None):

        if x.dim() > 2:
            x = x.view(-1, self.re_size * self.re_size)

        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z, c=None):

        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += num_labels

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):

        if self.conditional:
            c = idx2onehot(c, n=10)
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z, c):

        if self.conditional:
            c = idx2onehot(c, n=10)
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x
    
def idx2onehot(idx, n):

    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    
    return onehot


class convVAE(nn.Module):
    def __init__(self, latent_size, re_size=28,
                 c1_dim=8, c2_dim=16, lin_dim=16):
        super().__init__()

        self.l_size = re_size // 4
        # o = (i - k + 2p) / s + 1

        self.c1_dim = c1_dim
        self.c2_dim = c2_dim
        self.lin_dim = lin_dim
        
        self.conv1 = nn.Conv2d(1, c1_dim, kernel_size=3, stride=2, padding=1)                                               # output dim: 14 weights dimension: 10*1*3*3
        self.conv2 = nn.Conv2d(c1_dim, c2_dim, kernel_size=3, stride=2, padding=1)                                              # output dim: 7 weights dimension: 20*10*3*3
        self.fc1 = nn.Linear(self.l_size*self.l_size*c2_dim, lin_dim)
        self.fc21 = nn.Linear(lin_dim, latent_size)
        self.fc22 = nn.Linear(lin_dim, latent_size)

        self.fc3 = nn.Linear(latent_size, lin_dim)
        self.fc4 = nn.Linear(lin_dim, self.l_size*self.l_size*c2_dim)
        self.deconv1 = nn.ConvTranspose2d(c2_dim, c1_dim, kernel_size=3, stride=2, padding=1, output_padding=1)                 # output dim: 14
        self.deconv2 = nn.ConvTranspose2d(c1_dim, 1, kernel_size=3, stride=2, padding=1, output_padding=1)                  # output dim: 28

    def convEncoder(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.l_size*self.l_size*self.c2_dim)
        x = F.relu(self.fc1(x))
        means = self.fc21(x)
        log_var = self.fc22(x)
        return means, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def convDecoder(self, x):
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, self.c2_dim, self.l_size, self.l_size)
        x = F.relu(self.deconv1(x))
        x = torch.sigmoid(self.deconv2(x))
        return x

    def forward(self, x):
        means, log_var = self.convEncoder(x)
        z = self.reparameterize(means, log_var)
        recon_x = self.convDecoder(z)

        return recon_x, means, log_var, z
    
    def inference(self, z):
        recon_x = self.convDecoder(z)
        return recon_x
    

# network of AutoEncoder
class convAE(nn.Module):
    def __init__(self, latent_size, re_size=28):
        super().__init__()

        self.l_size = re_size // 4
        # o = (i - k + 2p) / s + 1
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=2, padding=1)  # output dim: 14 weights dimension: 10*1*3*3
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, stride=2, padding=1)  # output dim: 7 weights dimension: 20*10*3*3
        self.bn2 = nn.BatchNorm2d(20)
        self.fc21 = nn.Linear(self.l_size*self.l_size*20, latent_size)

        self.fc3 = nn.Linear(latent_size, self.l_size*self.l_size*20)
        self.bn4 = nn.BatchNorm1d(self.l_size*self.l_size*20)
        # self.fc4 = nn.Linear(30, 7*7*20)
        # self.bn5 = nn.BatchNorm1d(7*7*20)
        self.deconv1 = nn.ConvTranspose2d(20, 10, kernel_size=3, stride=2, padding=1, output_padding=1) # output dim: 14
        self.bn6 = nn.BatchNorm2d(10)
        self.deconv2 = nn.ConvTranspose2d(10, 1, kernel_size=3, stride=2, padding=1, output_padding=1) # output dim: 28

    def convEncoder(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(-1, self.l_size*self.l_size*20)
        latent = self.fc21(x)
        return latent

    def convDecoder(self, x):
        x = F.relu(self.bn4(self.fc3(x)))
        x = x.view(-1, 20, self.l_size, self.l_size)
        x = F.relu(self.bn6(self.deconv1(x)))
        x = torch.sigmoid(self.deconv2(x))
        return x

    def forward(self, x):
        latent = self.convEncoder(x)
        recon_x = self.convDecoder(latent)

        return recon_x, latent
    
    def inference(self, z):
        recon_x = self.convDecoder(z)
        return recon_x