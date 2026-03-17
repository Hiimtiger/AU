import torch
import torch.nn as nn
import torch.nn.functional as F

# Weight Standardized Conv2d
class WSConv2d(nn.Conv2d):
    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1, keepdim=True).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

# Basic Conv Block with Residual & LeakyReLU
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = WSConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = WSConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.LeakyReLU(inplace=True)

        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += identity
        out = self.act2(out)
        return out

# Attention Gate
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.GroupNorm(8, F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.GroupNorm(8, F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.GroupNorm(1, 1),
            nn.Sigmoid()
        )
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.act(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# Full Attention U-Net
class AttentionUNet(nn.Module):
    def __init__(self, img_ch, output_ch=1, dropout_rate=0.2):
        super(AttentionUNet, self).__init__()

        # Encoder
        self.conv1 = ConvBlock(img_ch, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = ConvBlock(256, 512)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.conv3_1 = ConvBlock(512, 256)
        self.drop3 = nn.Dropout2d(dropout_rate)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.conv2_1 = ConvBlock(256, 128)
        self.drop2 = nn.Dropout2d(dropout_rate)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.conv1_1 = ConvBlock(128, 64)
        self.drop1 = nn.Dropout2d(dropout_rate)

        self.final = nn.Conv2d(64, output_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        p1 = self.pool1(x1)

        x2 = self.conv2(p1)
        p2 = self.pool2(x2)

        x3 = self.conv3(p2)
        p3 = self.pool3(x3)

        x4 = self.conv4(p3)

        # Decoder
        d3 = self.up3(x4)
        x3 = self.att3(d3, x3)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.drop3(self.conv3_1(d3))

        d2 = self.up2(d3)
        x2 = self.att2(d2, x2)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.drop2(self.conv2_1(d2))

        d1 = self.up1(d2)
        x1 = self.att1(d1, x1)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.drop1(self.conv1_1(d1))

        out = self.final(d1)
        return out
