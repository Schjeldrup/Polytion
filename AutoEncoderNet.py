import torch
from Polytion import Generator as g

class Autoencoder(torch.nn.Module):
    def __init__(self, layer, N, rank, bottleneck_dim, HR_dim, downscalefactor, scalefactor, layerOptions, generatorOptions):
        super(Autoencoder,self).__init__()
        self.encoder = g.Generator(layer, N, rank, bottleneck_dim, bottleneck_dim, downscalefactor, layerOptions, generatorOptions)
        self.decoder = g.Generator(layer, N, rank, HR_dim, HR_dim, scalefactor, layerOptions, generatorOptions)

    def forward(self, x):
        x = self.encoder(x.float())
        x = self.decoder(x)
        return x

class Autoencoder_seq(torch.nn.Module):
    def __init__(self, layer, N, rank, bottleneck_dim, HR_dim, downscalefactor, scalefactor, layerOptions, generatorOptions):
        super(Autoencoder_seq,self).__init__()
        self.encoder = g.Generator_seq(layer, N, rank, bottleneck_dim, bottleneck_dim, downscalefactor, layerOptions, generatorOptions)
        self.decoder = g.Generator_seq(layer, N, rank, HR_dim, HR_dim, scalefactor, layerOptions, generatorOptions)

    def forward(self, x):
        x = self.encoder(x.float())
        x = self.decoder(x)
        return x