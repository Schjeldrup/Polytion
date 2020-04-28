# Here we can gather some different image quality assesment loss functions
import torch
import torch.nn.functional as F
from math import log10, exp, sqrt

def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# See https://mkfmiku.github.io/loss-functions-in-image-enhancement/
class PSNRLoss(torch.nn.Module):
    def __init__(self):
        super(PSNRLoss, self).__init__()


    def forward(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return self.tvloss_weight * (h_tv + w_tv) / (b * c * h * w)


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


# See https://mkfmiku.github.io/loss-functions-in-image-enhancement/
class TVLoss(torch.nn.Module):
    def __init__(self, tvloss_weight=1.e-9):
        super(TVLoss, self).__init__()
        self.tvloss_weight = tvloss_weight

    def forward(self, generated):
        b, c, h, w = generated.size()
        h_tv = torch.pow((generated[:, :, 1:, :] - generated[:, :, :(h - 1), :]), 2).sum()
        w_tv = torch.pow((generated[:, :, :, 1:] - generated[:, :, :, :(w - 1)]), 2).sum()
        return self.tvloss_weight * (h_tv + w_tv) / (b * c * h * w)
