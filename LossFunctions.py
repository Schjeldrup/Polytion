# Here we can gather some different image quality assesment loss functions
import torch
import torch.nn.functional as F
from math import log10, exp, sqrt
import numpy as np
from skimage.metrics import structural_similarity

class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, image1, image2):
        return torch.mean(torch.sqrt((image1 - image2)**2 + self.eps))

def smooth_l1(input, target):
    t = np.abs(input - target)
    return np.mean(np.where(t < 1, 0.5 * (input - target) ** 2, t - 0.5))


# See https://mkfmiku.github.io/loss-functions-in-image-enhancement/
# See also Remi Flamary for how to use L2 and TV together
class TVLoss(torch.nn.Module):
    def __init__(self, tvloss_weight=1e-5):
        super(TVLoss, self).__init__()
        self.tvloss_weight = tvloss_weight

    def forward(self, image):
        b, c, h, w = image.size()
        h_tv = torch.pow((image[:, :, 1:, :] - image[:, :, :(h - 1), :]), 2).sum()
        w_tv = torch.pow((image[:, :, :, 1:] - image[:, :, :, :(w - 1)]), 2).sum()
        return self.tvloss_weight * torch.sqrt(h_tv + w_tv)


class StyleLoss(torch.nn.Module):
    def __init__(self, StyleLossWeight=1e-5):
        super(StyleLoss, self).__init__()
        self.StyleLossWeight = StyleLossWeight

    def forward(self, im1, im2):
        b, c, w, h = im1.shape
        #img1 = im1.reshape(b, c, w*h)
        #img2 = im2.reshape(b, c, w*h)

        #bgram1 = torch.mm(im1.reshape(b, hw), im1.reshape(hw, b))
        #bgram2 = torch.mm(im2.reshape(b, hw), im2.reshape(hw, b))
        #bgram1 = torch.mm(img1, img1.t())
        #bgram2 = torch.mm(img2, img2.t())
        bgram1 = torch.bmm(im1.view(b, 1, w*h), im1.view(b, w*h, 1))
        bgram2 = torch.bmm(im2.view(b, 1, w*h), im2.view(b, w*h, 1))
        #print(bgram1.shape)
        #print(((bgram1 - bgram2) ** 2).shape)

        return self.StyleLossWeight * torch.mean((bgram1 - bgram2) ** 2)


class StyleLossOld(torch.nn.Module):
    def __init__(self, StyleLossWeight=1e-5):
        super(StyleLoss, self).__init__()
        self.StyleLossWeight = StyleLossWeight

    def forward(self, im1, im2):
        bgram1 = torch.bmm(im1, im1.permute(0, 2, 1))
        bgram2 = torch.bmm(im2, im2.permute(0, 2, 1))
        return self.StyleLossWeight * torch.mean((bgram1 - bgram2) ** 2)



# Orthogonal loss: enforce orthogonal matrices in the weights
class OrthLoss(torch.nn.Module):
    def __init__(self, orthloss_weight=1e-5):
        super(OrthLoss, self).__init__()
        self.orthloss_weight = orthloss_weight

    def forward(self, model, device):
        loss = torch.tensor(0., dtype=torch.float32, device=device)
        for name, param in model.named_parameters():
            if 'TT' in name and param.requires_grad:
                r1, s, r2 = param.shape
                r = r1 if r1 > r2 else r2
                #param, _ = torch.qr(param, some=False)
                #print(param.shape)
                #nodiag = torch.ones(r, s, s, dtype=torch.float32) - torch.eye(s, dtype=torch.float32)
                # print(param.permute(0, 2, 1).shape)
                # print(param.permute(0, 1, 2).shape)
                #print(torch.bmm(param.permute(0, 2, 1), param.permute(0, 1, 2)).shape)
                # weight_squared = torch.bmm(param.permute(0, 1, 2), param.permute(0, 2, 1))
                weight_squared = torch.bmm(param.permute(0, 2, 1), param.permute(0, 1, 2))
                weight_squared[:, range(r2), range(r2)] -= 1
                loss += torch.sqrt((weight_squared ** 2)).sum()
        return self.orthloss_weight * loss


# See https://mkfmiku.github.io/loss-functions-in-image-enhancement/
class PSNRLoss(torch.nn.Module):
    def __init__(self):
        super(PSNRLoss, self).__init__()
    def forward(self, img1, img2):
        mse = torch.mean((img1 -img2) ** 2) + 1.0e-7
        out = 10 * torch.log10( (torch.max(img1) - torch.min(img1))**2 / torch.sqrt(mse))
        return out.float()

# See https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html
class SkiMageSSIMLoss(torch.nn.Module):
    def __init__(self):
        super(SkiMageSSIMLoss, self).__init__()
    def forward(self, img1, img2):
        print(img1.shape)
        b, c, w, h = img1.shape
        ssim = 0
        for i in range(b):
            im1 = img1[i].reshape(w,h).detach().numpy()
            im2 = img2[i].reshape(w,h).detach().numpy()
            datrange = im1.max() - im2.min()
            ssim += structural_similarity(im1, im2, data_range=datrange, multichannel=True)
        return torch.tensor(ssim/b)


# See https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def ssim(self, img1, img2, window, window_size, channel, size_average = True):
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2
        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return self.ssim(img1, img2, window, self.window_size, channel, self.size_average)

