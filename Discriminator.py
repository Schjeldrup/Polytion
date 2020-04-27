import torch
import torch.nn as nn

#Takes a low-resolution pic and a high-resolution and outputs a probability of it being a fake
class discriminator(nn.Module):
    def __init__(self,batch_size):
        super(discriminator, self).__init__()
        
        # On the {z_s,d}
        self.disc = nn.Sequential(
        nn.Conv2d(1, 16 , kernel_size=3, stride=1,padding=1),
        nn.LeakyReLU(0.2)
        )
        
        self.disc2 = nn.Sequential(
        nn.Conv2d(2, 2 , kernel_size=3, stride=1,padding=0),
        nn.LeakyReLU(0.2),
        nn.Conv2d(2, 4 , kernel_size=1, stride=1,padding=0),
        nn.LeakyReLU(0.2),

        nn.Conv2d(4, 4 , kernel_size=3, stride=1,padding=0),
        nn.LeakyReLU(0.2),
        nn.Conv2d(4, 8 , kernel_size=3, stride=1,padding=0),
        nn.LeakyReLU(0.2),
        
        nn.Conv2d(8, 8 , kernel_size=3, stride=1,padding=0),
        nn.LeakyReLU(0.2),
        nn.Conv2d(8, 1 , kernel_size=3, stride=1,padding=0),
        nn.LeakyReLU(0.2)
        )
        
        self.disc3 = nn.Sequential(
        nn.Linear(in_features=252004, out_features=1, bias=True)#,
        #nn.Sigmoid()
        )
        
        self.sig = nn.Sequential(
        #nn.Linear(in_features=252004, out_features=1, bias=True)#,
        nn.Sigmoid()
        )
    def forward(self, lowres,highres):
        upscaledlow=self.disc(lowres)
        upscaledlow=upscaledlow.view(highres.size())
        combine= torch.cat([upscaledlow,highres], dim=1)
        res=self.disc2(combine)
        res=res.view(highres.size()[0],252004)
        res=self.disc3(res)
        res*=100
        res=self.sig(res)
        return res