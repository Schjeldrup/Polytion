import torch 
import time
import pickle

# Load parent folders
import sys
sys.path.insert(0, '..')
from Polytion import Generator as g
from Polytion import AutoEncoderNet as AE
import matplotlib.pyplot as plt

# Get the required images: (change for real testfolder later!!)
imagefolderpath = '000001_01_01'
LRpath = imagefolderpath + '/LRimages.pickle'
HRpath = imagefolderpath + '/HRimages.pickle'
try:
    with open(LRpath, 'rb') as handle:
        LRimages = pickle.load(handle)
    with open(HRpath, 'rb') as handle:
        HRimages = pickle.load(handle)
except ImportError:
    print("Image files don't exist")

# Load the model and parameters:
modeltime = "04-05-2020_10:05:13"
modelname = "Testing/Trained_CP_model.pth"
modelparamsname = "Testing/Trained_CP_model_params.pth"

mp = torch.load(modelparamsname)
model = AE.Autoencoder_seq(mp["layer"], mp["N"], mp["rank"], mp["bottleneck_dim"], mp["HR_dim"], mp["downscalefactor"], mp["scalefactor"], mp["layerOptions"], mp["generatorOptions"])
model.load_state_dict(torch.load(modelname))

# Test:
index = 1
model.eval()
test = torch.tensor(LRimages[index]).reshape(1,1,mp["LR_dim"],mp["LR_dim"])
output = model(test).reshape(mp["HR_dim"], mp["HR_dim"]).detach().numpy()


# Plot example:
fig, ax = plt.subplots(1,3, figsize=(15,5))
ax[0].imshow(LRimages[index], cmap='gray')
ax[0].set_title("Input")
ax[0].axis('off')

ax[1].imshow(output, cmap='gray')
ax[1].set_title("Output")
ax[1].axis('off')

ax[2].imshow(HRimages[index], cmap='gray')
ax[2].set_title("Truth")
ax[2].axis('off')

# Uncomment to save output: result has same timestamp as tested model
filename = "Testing/TestingResult_" + modeltime + ".png"
fig.savefig(filename, bbox_inches='tight')
plt.show()

