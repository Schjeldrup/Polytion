import torch
import Generator as gen
import prepData as prep

# Parameters:
N = 5
rank = 4
scalefactor = 4
HiRes_imwidth, HiRes_imheight = 512, 512
LoRes_imwidth, LoRes_imheight = 128, 128

batch_size = 5

# Give the images:
images = prep.load_images_from_folder('000001_01_01')
HiResImages = prep.normalize(images)
LoResImages = prep.compress_images(images)

HiResLoader = torch.utils.data.DataLoader(HiResImages, batch_size=batch_size)
LoResLoader = torch.utils.data.DataLoader(LoResImages, batch_size=batch_size)

# Model and optimization
num_epochs = 1

class Autoencoder(torch.nn.Module):
    def __init__(self, layer):
        super(Autoencoder,self).__init__()

    #def encoder(self, ):

    #def decoder(self, ):

        #self.encoder = layer(N, rank, imwidth, imheight)
        #self.decoder = layer(N, rank, imwidth, imheight)
        self.encoder = gen.Generator(gen.FTT_Layer, N, rank, LoRes_imwidth, LoRes_imheight, 0.5)
        self.decoder = gen.Generator(gen.FTT_Layer, N, rank, HiRes_imwidth, HiRes_imwidth, scalefactor)

    def forward(self, x):
        print("Input shape = ", x.shape)
        x = self.encoder(x.float())
        print("After encoder shape = ", x.shape)
        #x = self.decoder(x)
        return x

model = Autoencoder(gen.FTT_Layer)
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

# Training:
for epoch in range(num_epochs):
    for img in LoResLoader:
        print("Image shape =")
        print(img.shape)
        img = torch.autograd.Variable(img.reshape(batch_size, 1, LoRes_imwidth, LoRes_imheight))
        output = model(img)
        print("Image shape =")
        print(img.shape)
        break
        loss = distance(output, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.data()))
