import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.leaky_relu(self.fc1(w))
        w = self.sigmoid(self.fc2(w))
        return x * w

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(CBAM, self).__init__()
        self.channel_attention = SEBlock(in_channels, reduction)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.channel_attention(x)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        pool = torch.cat([max_pool, avg_pool], dim=1)
        return x * self.spatial_attention(pool)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_cbam=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.LeakyReLU = nn.LeakyReLU(inplace=True)
        self.cbam = CBAM(out_channels) if use_cbam else None

    def forward(self, x):
        x = self.LeakyReLU(self.bn(self.conv(x)))
        if self.cbam:
            x = self.cbam(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, use_cbam=False):
        super(Encoder, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, use_cbam=use_cbam)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_block(x)
        return x, self.pool(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, use_cbam=False):
        super(Decoder, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, use_cbam=use_cbam)
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x, skip):
        x = torch.cat((self.upsample(x), skip), dim=1)
        return self.conv_block(x)

class Decoder_Bilinear(nn.Module):
    def __init__(self, in_channels, out_channels, use_cbam=False):
        super(Decoder_Bilinear, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.channel_adjust = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_block = ConvBlock(in_channels, out_channels, use_cbam=use_cbam)

    def forward(self, x, skip):
        x = self.channel_adjust(self.upsample(x))
        x = torch.cat((x, skip), dim=1)
        return self.conv_block(x)