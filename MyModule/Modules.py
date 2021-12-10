import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

#%% Basic Architexture
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = EncoderV1()
    def forward(self,x):
        code = self.encoder(x)
        return code

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = DecoderV1()
    def forward(self,code):
        x_recon = self.decoder(code)
        return x_recon


#%% Encoders
class EncoderV1(nn.Module):
    def __init__(self):
        super(EncoderV1, self).__init__()
        firstConvChannel = 4
        outChannel = 4
        layers = []
        layers.append(nn.ReflectionPad2d(padding=(7,7,0,0))) #(N,2,8,50)->(N,2,8,64)

        # Initial conv block   (N,2,8,64)->(N,4,8,64)
        layers += [
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_channels=2,out_channels=firstConvChannel,kernel_size=5),
            nn.InstanceNorm2d(num_features=firstConvChannel),
            nn.ReLU(inplace=True),
        ]

        #Downsampling
        for _ in range(2): #(N,4,8,64) -> (N,16,4,32) -> (N,64,2,16)
            layers += [
                nn.Conv2d(in_channels=firstConvChannel,out_channels=firstConvChannel * 4,
                          kernel_size= 4,stride=2,padding= 1),
                nn.InstanceNorm2d(num_features=firstConvChannel * 4),
                nn.ReLU(inplace=True),
            ]
            firstConvChannel *= 4

        #Residual blocks #(N,64,2,16) -> (N,64,2,16)
        for _ in range(4):
            layers += [ResidualBlock2d(features=firstConvChannel,norms='instance')]

        #output layer #(N,64,2,16)->(N,4,2,16) -> (N,4*2*16)
        layers += [
            nn.Conv2d(in_channels=firstConvChannel,out_channels=outChannel,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1)
        ]
        self.model = nn.Sequential(*layers)
    def forward(self,x):
        return self.model(x) #((N,2,8,50) -> (N,4*2*16)

class DecoderV1(nn.Module):
    def __init__(self):
        super(DecoderV1, self).__init__()
        layers = []
        dim = 64
        #unflatten # (N,4*2*16) -> (N,4,2,16)
        layers +=[
            nn.Unflatten(dim=1,unflattened_size=(4,2,16))
        ]
        #init input (N,4,2,16) -> (N,64,2,16)
        layers += [
            nn.Conv2d(in_channels=4,out_channels= dim,kernel_size=1),
            nn.ReLU(inplace=True)
        ]

        #Residual blocks #(N,64,2,16) -> (N,64,2,16)
        for _ in range(4):
            layers += [ResidualBlock2d(features=dim,norms='instance')]

        #Upsampling (N,64,2,16) -> (N,16,4,32) -> (N,4,8,64)
        for _ in range(2):
            layers += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels=dim,out_channels=dim//4,kernel_size=5,padding=2),
                nn.InstanceNorm2d(dim//4),
                nn.ReLU(inplace=True),
            ]
            dim = dim // 4
        # Output layer (N,4,8,64) -> (N,2,8,64) -> (N,2,8,50)
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=dim,out_channels=2,kernel_size=7),
            nn.Tanh(),
            nn.AdaptiveAvgPool2d(output_size=(8,50)),
        ]
        self.model = nn.Sequential(*layers)
    def forward(self,range_code):
        x_recon = self.model(range_code)
        return x_recon



#%% Tool layer
class ResidualBlock2d(nn.Module):
    def __init__(self,features,norms = "instance"):
        super(ResidualBlock2d, self).__init__()
        normLayer = nn.InstanceNorm2d if norms == "instance" else nn.BatchNorm2d

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features,features,3),
            normLayer(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features,features,3),
            normLayer(features),
        )
    def forward(self,x):
        return x + self.block(x)
