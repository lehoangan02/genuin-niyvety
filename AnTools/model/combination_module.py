import torch.nn.functional as F
import torch as nn
import torch

class NoPromptCombinationModule(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1),
                                    nn.BatchNorm2d(c_out),
                                    nn.LeakyReLU(0.01, inplace=True))

        self.conv2concat = nn.Sequential(nn.Conv2d(c_out*2, c_out, kernel_size=3, padding=1, stride=1),
                                        nn.BatchNorm2d(c_out),
                                        nn.LeakyReLU(0.01, inplace=True))
    def forward(self, x_dense, x_sparse):
        x_sparse_upsampled = F.interpolate(x_sparse, size=x_dense.shape[-2:], mode='bilinear', align_corners=False)
        x = self.conv1(x_dense)
        x = torch.cat((x, x_sparse_upsampled), dim=1)
        x = self.conv2concat(x)
        return x