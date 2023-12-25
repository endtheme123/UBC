import numpy as np
import torch
from torch import nn
from torchvision.models.resnet import resnet18
from torchvision import transforms
import copy 

class UBC(nn.Module):

    def __init__(self, img_size, nb_channels, latent_img_size, z_dim, rec_loss="xent", beta=1, delta=1):
        '''
        '''
        super(UBC, self).__init__()

        self.img_size = img_size
        self.nb_channels = nb_channels
        self.latent_img_size = latent_img_size
        self.z_dim = z_dim
        self.beta = beta
        self.rec_loss = rec_loss
        self.delta = delta
        self.nb_conv = 3
        # the depth we will have at the end of the encoder given that a
        # convolution incease depth by 2 starting at 32 after the first
        self.max_depth_conv = 2 ** (4 + self.nb_conv)
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        for param in self.dino.parameters():
            param.requires_grad = False

        # self.dino.norm = nn.Identity()
        # self.dino.head = nn.Identity()
        
        # self.dino_export_loc = nn.Sequential(
        #     nn.Conv2d(384,384, 4, 2, 0),
            
        #     nn.ReLU()
            
        # )
        # self.dino_export_glo = copy.deepcopy(self.dino_export_loc)
        self.resnet = resnet18(pretrained=False)
        self.resnet_entry = nn.Sequential(
            nn.Conv2d(self.nb_channels, 64, kernel_size=7,
                stride=2, padding=3, bias=False),
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )
        self.resnet18_layer_list = [
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4 
        ]
        self.encoder_layers = [self.resnet_entry] 
        for i in range(1, self.nb_conv): 
            try:
                self.encoder_layers.append(self.resnet18_layer_list[i - 1])
            except IndexError: 
                depth_in = 2 ** (4 + i)
                depth_out = 2 ** (4 + i + 1)
                self.encoder_layers.append(nn.Sequential(
                    nn.Conv2d(depth_in, depth_out, 4, 2, 1),
                    nn.BatchNorm2d(depth_out),
                    nn.ReLU()
                    ))
        self.conv_encoder = nn.Sequential(
            *self.encoder_layers,
        )
        self.final_encoder = nn.Sequential(
            nn.Conv2d(self.max_depth_conv, self.z_dim * 2, kernel_size=1,
            stride=1, padding=0)
        )

        # self.glb_conv_encoder = copy.deepcopy(self.conv_encoder)
        # self.glb_final_encoder = copy.deepcopy(self.final_encoder)

        # self.initial_decoder = nn.Sequential(
        #     nn.ConvTranspose2d(self.z_dim, self.max_depth_conv,
        #         kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(self.max_depth_conv),
        #     nn.ReLU()
        # )
            
        nb_conv_dec = self.nb_conv

        self.decoder_layers = []
        self.decoder_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(192, 96 , 7, 1, 0),
                ))
        self.decoder_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(96, 48, 4, 2, 1),
                    nn.BatchNorm2d(48),
                    nn.ReLU()
                ))
        
        self.decoder_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(48, 24, 4, 2, 1),
                    nn.BatchNorm2d(24),
                    nn.ReLU()
                ))
        
        self.decoder_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(24, 12, 4, 2, 1),
                    nn.BatchNorm2d(12),
                    nn.ReLU()
                ))
        self.decoder_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(12, 6, 4, 2, 1),
                    nn.BatchNorm2d(6),
                    nn.ReLU()
                ))
        self.decoder_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(6, 3, 4, 2, 1),
                    nn.BatchNorm2d(3),
                    nn.ReLU()
                ))
        # for i in reversed(range(nb_conv_dec)):
        #     depth_in = 2 ** (4 + i + 1)
        #     depth_out = 2 ** (4 + i)
        #     if i == 0:
        #         depth_out = self.nb_channels
        #         self.decoder_layers.append(nn.Sequential(
        #             nn.ConvTranspose2d(depth_in, depth_out, 4, 2, 1),
        #         ))
        #     else:
        #         self.decoder_layers.append(nn.Sequential(
        #             nn.ConvTranspose2d(depth_in, depth_out, 4, 2, 1),
        #             nn.BatchNorm2d(depth_out),
        #             nn.ReLU()
        #         ))
        self.conv_decoder = nn.Sequential(
            *self.decoder_layers
        )


    def encoder(self, x, x_glb):
        x = self.dino(x)
        # print(x.shape)
        # x = self.dino_export_loc(x)
        # x = x.view(4, 384,14,14)
        x_glb = self.dino(x_glb)
        # x_glb = self.dino_export_glo(x_glb)
        # x_glb = x_glb.view(4, 384,14,14)
        return 0.75*x[:, :self.z_dim] + 0.25*x_glb[:, :self.z_dim], 0.75*x[:, self.z_dim:] + 0.25*x_glb[:, self.z_dim:]

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(torch.mul(logvar, 0.5))
            eps = torch.randn_like(std)
            # print(mu.shape)
            # print(logvar.shape)
            return eps * std + mu
        else:
            return mu

    def decoder(self, z):
        # z = self.initial_decoder(z)
        x = self.conv_decoder(z)
        x = nn.Sigmoid()(x)
        return x

    def forward(self, x, x_glb):
        mu, logvar = self.encoder(x, x_glb)
        # print(mu.shape)
        
        mu = torch.unsqueeze(mu, -1)
        mu = torch.unsqueeze(mu, -1)
        logvar = torch.unsqueeze(logvar, -1)
        logvar = torch.unsqueeze(logvar, -1)
        
        self.mu = mu
        self.logvar = logvar
        z = self.reparameterize(mu, logvar)
        # print(z.shape)
        
        # print(mu.shape)
        # print(self.decoder(z).shape)
        return self.decoder(z), (mu, logvar)

    def xent_continuous_ber(self, recon_x, x, pixelwise=False):
        ''' p(x_i|z_i) a continuous bernoulli '''
        eps = 1e-6
        # print(recon_x.shape)
        # print(x.shape)
        recon = transforms.Resize(size=(256, 256))
        def log_norm_const(x):
            # numerically stable computation
            x = torch.clamp(x, eps, 1 - eps)
            x = torch.where((x < 0.49) | (x > 0.51), x, 0.49 *
                    torch.ones_like(x))
            return torch.log((2 * self.tarctanh(1 - 2 * x)) /
                            (1 - 2 * x) + eps)
        if pixelwise:
            return (x * torch.log(recon_x + eps) +
                            (1 - x) * torch.log(1 - recon_x + eps) +
                            log_norm_const(recon_x))
        else:
            return torch.sum(x * torch.log(recon_x + eps) +
                            (1 - x) * torch.log(1 - recon_x + eps) +
                            log_norm_const(recon_x), dim=(1, 2, 3))

    def mean_from_lambda(self, l):
        ''' because the mean of a continuous bernoulli is not its lambda '''
        l = torch.clamp(l, 10e-6, 1 - 10e-6)
        l = torch.where((l < 0.49) | (l > 0.51), l, 0.49 *
            torch.ones_like(l))
        return l / (2 * l - 1) + 1 / (2 * self.tarctanh(1 - 2 * l))

    def kld(self):
        # NOTE -kld actually
        return 0.5 * torch.sum(
                1 + self.logvar - self.mu.pow(2) - self.logvar.exp(),
            dim=(1)
        )

    def loss_function(self, recon_x, x):
        rec_term = self.xent_continuous_ber(recon_x, x)
        rec_term = torch.mean(rec_term)

        kld = torch.mean(self.kld())

        L = (rec_term + self.beta * kld)

        loss = L

        loss_dict = {
            'loss': loss,
            'rec_term': rec_term,
            '-beta*kld': self.beta * kld
        }

        return loss, loss_dict

    def step(self, input_mb_loc, input_mb_glo):
        recon_mb, _ = self.forward(input_mb_loc, input_mb_glo)

        loss, loss_dict = self.loss_function(recon_mb, input_mb_loc)

        recon_mb = self.mean_from_lambda(recon_mb)

        return loss, recon_mb, loss_dict

    def tarctanh(self, x):
        return 0.5 * torch.log((1+x)/(1-x))

        
