import torch.nn.functional as F
import torch.nn as nn
import torch

class NoPromptUpScaleModule(nn.Module):
    def __init__(self, c_low, c_high, c_out):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(c_low, c_high, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(c_high),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.conv2concat = nn.Sequential(
            nn.Conv2d(c_high + c_high, c_out, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.01, inplace=True)
        )

    def forward(self, low_res, high_res):
        low_res_up = F.interpolate(low_res, size=high_res.shape[-2:], mode='bilinear', align_corners=True)
        low_res_up = self.conv1(low_res_up)
        x = torch.cat((high_res, low_res_up), dim=1)
        x = self.conv2concat(x)
        return x

