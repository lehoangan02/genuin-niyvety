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
class PromptUpScaleModuleV1(nn.Module):
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
        self.conv3 = nn.Sequential(
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.firstDepthSepConv = DepthwiseSeparableConv(
            in_channels=c_out,
            out_channels=c_out,
            custom_depthwise_weight=None  # to be set later
        )


    def forward(self, low_res, high_res, filter_prompt):
        low_res_up = F.interpolate(low_res, size=high_res.shape[-2:], mode='bilinear', align_corners=True)
        low_res_up = self.conv1(low_res_up)
        x = torch.cat((high_res, low_res_up), dim=1)
        x = self.conv2concat(x)
        # convolve the concatenated feature map with the filter prompt
        self.firstDepthSepConv.depthwise_weight = filter_prompt
        x = self.firstDepthSepConv(x)
        x = self.conv3(x)
        return x

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, custom_depthwise_weight=None, stride=1, padding=1):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        if custom_depthwise_weight is not None:
            assert custom_depthwise_weight.shape == (in_channels, 1, 3, 3)
            self.register_buffer('depthwise_weight', custom_depthwise_weight)
        else:
            self.register_buffer('depthwise_weight', None)

    def forward(self, x):
        assert self.depthwise_weight is not None, "depthwise_weight not set"
        
        # x shape: [B, C, H, W] (e.g., [8, 128, 72, 128])
        # weight shape: [B, C, 1, kH, kW] (e.g., [8, 128, 1, 3, 3])
        
        B, C, H, W = x.shape
        kH, kW = self.depthwise_weight.shape[-2:]
        
        # 1. Reshape input x from [B, C, H, W] to [1, B*C, H, W]
        x_reshaped = x.reshape(1, B * C, H, W)
        
        # 2. Reshape weight from [B, C, 1, k, k] to [B*C, 1, k, k]
        w_reshaped = self.depthwise_weight.reshape(B * C, 1, kH, kW)
        
        # 3. Apply depthwise conv
        # groups=B*C ensures each of the B*C channels is convolved with its own filter
        out_depthwise = F.conv2d(x_reshaped, w_reshaped, stride=self.stride, padding=self.padding, groups=B * C)
        
        # 4. Reshape output back to [B, C, H_out, W_out]
        _, _, H_out, W_out = out_depthwise.shape
        x = out_depthwise.reshape(B, C, H_out, W_out)
        
        # 5. Apply pointwise conv (which works on batches)
        x = self.pointwise(x)
        return x
class FirstFilterGeneratorV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3072, 512, kernel_size=3, stride=2, padding=1),  # [B, 512, 7, 7]
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),   # [B, 256, 4, 4]
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(256, 128, kernel_size=2, stride=1, padding=0),   # [B, 128, 3, 3]
        )
    
    def forward(self, x):
        # x is [B, 3072, 14, 14]
        # The reshape line is REMOVED
        x = self.net(x)  # Output shape: [B, 128, 3, 3]
        
        # We want [B, 128, 1, 3, 3] for the depthwise conv weight
        # The old "squeeze/unsqueeze" was only for B=1
        return x.unsqueeze(2)  # Output shape: [B, 128, 1, 3, 3]
class SecondFilterGeneratorV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # CHANGED: stride=2 to stride=1 to maintain 3x3 spatial dimensions
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # [B, 64, 3, 3]
            nn.LeakyReLU(0.01, inplace=True)
        )
    
    def forward(self, x):
        # x input shape is [B, 128, 1, 3, 3]
        
        # Squeeze dim 2 to make it a 4D batch: [B, 128, 3, 3]
        x = x.squeeze(2)
        
        # Pass through the conv net
        x = self.net(x)  # Output shape: [B, 64, 3, 3]
        
        # Unsqueeze dim 2 to re-format for next depthwise conv
        # Final shape: [B, 64, 1, 3, 3]
        return x.unsqueeze(2)