# In this script we will be training and testing the AutoEncoder with multiple
# loss functions and parameter setups.

# Get access to parent folders
import sys
sys.path.insert(0, '..')

import torch
import torchvision

import matplotlib.pyplot as plt
import numpy as np
import time
import tqdm

import Generator as g
import prepData as prep
import ImageQualityAssesment as iqa

cuda = torch.cuda.is_available()
if cuda:
    print("cuda session enabled")
    device = torch.device("cuda:0")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print("gpu session enabled")
    device = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')

# Parameters:
batch_size = 15
N = 6
rank = 10

LR_dim = 128
HR_dim = 512
bottleneck_dim = 32

scalefactor = HR_dim/bottleneck_dim
downscalefactor = bottleneck_dim/LR_dim

images = prep.load_images_from_folder('../000001_01_01')
HRimages = prep.normalize_0(images)
LRimages = prep.compress_images(HRimages)

#HR_loader = torch.utils.data.DataLoader(HRimages,batch_size=batch_size, pin_memory=cuda)
#LR_loader = torch.utils.data.DataLoader(LRimages, batch_size=batch_size, pin_memory=cuda)
HR_loader = torch.utils.data.DataLoader(HRimages[:20],batch_size=batch_size, pin_memory=cuda)
LR_loader = torch.utils.data.DataLoader(LRimages[:20], batch_size=batch_size, pin_memory=cuda)

class Autoencoder(torch.nn.Module):
    def __init__(self, layer, layerOptions, generatorOptions):
        super(Autoencoder,self).__init__()
        self.encoder = g.Generator(layer, N, rank, bottleneck_dim, bottleneck_dim, downscalefactor, layerOptions, generatorOptions)
        self.decoder = g.Generator(layer, N, rank, HR_dim, HR_dim, scalefactor, layerOptions, generatorOptions)

    def forward(self, x):
        x = self.encoder(x.float())
        x = self.decoder(x)
        return x

MSE_lossfunc = torch.nn.MSELoss()
TV_lossfunc = iqa.TVLoss()
SSIM_lossfunc = iqa.SSIMLoss()

num_epochs = 10
def train(model):
    model.train()
    if cuda:
        model = model.cuda()

    epoch_loss = []
    all_loss = []
    MSE = []
    TV = []
    SSIM = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=0.001)

    epochs = tqdm.trange(num_epochs, desc='Bar desc', leave=True)
    epochs.set_description("Start training")
    epochs.refresh()
    try:
        for epoch in epochs:
            batch_loss = []
            for HiResIm, LoResIm in zip(HR_loader, LR_loader):
                HiResIm = HiResIm.unsqueeze_(1).float()
                LoResIm = LoResIm.unsqueeze_(1).float()
                HiResIm = torch.autograd.Variable(HiResIm).to(device)
                LoResIm = LoResIm.to(device)

                output = model(LoResIm).float()

                MSE_loss = MSE_lossfunc(output, HiResIm).float()
                TV_loss = TV_lossfunc(output).float()
                SSIM_loss = SSIM_lossfunc(output, HiResIm).float()
                loss = (1 - SSIM_loss)
                optimizer.zero_grad()
                loss.backward(retain_graph = True)
                MSE_loss.backward(retain_graph = True)
                MSE.append(MSE_loss.item())
                TV_loss.backward(retain_graph = True)
                TV.append(TV_loss.item())
                SSIM_loss.backward(retain_graph = True)
                SSIM.append(SSIM_loss.item())

                optimizer.step()
                lossvalue = loss.item()
                all_loss.append(lossvalue)
                batch_loss.append(lossvalue)

                #epochs.set_description("Loss = {:.2e}".format(lossvalue))
                epochs.set_description("Loss = " + str(lossvalue))
                epochs.refresh()

            epoch_loss.append(np.mean(batch_loss))

    except (KeyboardInterrupt, SystemExit):
        print("\nscript execution halted ..")
        print("loss = ", all_loss)
    print("training finished")
    return epoch_loss


# ## 3. Training the different layers and generators:
generatorOptions = {'parallel':True, 'workers':3}
layerOptions = {'randnormweights':False, 'normalize':False, 'parallel':True}

#PolyganCPlayer:
# model = Autoencoder(g.PolyganCPlayer, layerOptions, generatorOptions)
#
# epoch_loss = train(model)
# plt.plot(epoch_loss)
# plt.savefig('PolyganCPlayer.png', bbox_inches='tight')
#
# model.eval()
# test = torch.tensor(LRimages[0]).reshape(1,1,LR_dim,LR_dim)
# output = model(test).reshape(HR_dim,HR_dim)
# torchvision.utils.save_image(output, "outputPolyganCPlayer.jpg")

# PolyclassFTTlayer:
model = Autoencoder(g.PolyclassFTTlayer, layerOptions, generatorOptions)
epoch_loss = train(model)
plt.plot(epoch_loss)
plt.savefig('PolyclassFTTlayer.png', bbox_inches='tight')

model.eval()
test = torch.tensor(LRimages[0]).reshape(1,1,LR_dim,LR_dim)
output = model(test).reshape(HR_dim,HR_dim)
torchvision.utils.save_image(output, "outputPolyclassFTTlayer.jpg")
