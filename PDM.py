import torch
import torch.nn as nn
from Blocks import Encoder, ConvBlock, Decoder, Decoder_Bilinear

class PhysicalDehazingModule(nn.Module):
    """
    PDM: Physical Dehazing Module (即原 TA 网络结构)
    估计介质透射率 t 与大气/水下背景光 A。
    """
    def __init__(self):
        super(PhysicalDehazingModule, self).__init__()
        # 透射率网络分支
        self.encoder1 = Encoder(3, 32, use_cbam=True)
        self.encoder2 = Encoder(32, 64, use_cbam=True)
        self.encoder3 = Encoder(64, 128, use_cbam=True)
        self.encoder4 = Encoder(128, 256, use_cbam=True)

        self.middle_conv1 = ConvBlock(256, 512, use_cbam=True)

        self.decoder1 = Decoder(512, 256, use_cbam=True)
        self.decoder2 = Decoder(256, 128, use_cbam=True)
        self.decoder3 = Decoder(128, 64, use_cbam=True)
        self.decoder4 = Decoder_Bilinear(64, 32, use_cbam=True)

        self.final_conv1 = nn.Conv2d(32, 3, kernel_size=1)

        # 背景光估计分支
        self.middle_conv2 = ConvBlock(256, 512, use_cbam=True)

        self.output1 = ConvBlock(512, 256, use_cbam=True)
        self.output2 = ConvBlock(256, 128, use_cbam=True)
        self.output3 = ConvBlock(128, 64, use_cbam=True)
        self.output4 = ConvBlock(64, 32, use_cbam=True)

        self.final_conv2 = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x):
        skip1, x1 = self.encoder1(x)
        skip2, x1 = self.encoder2(x1)
        skip3, x1 = self.encoder3(x1)
        skip4, x1_ = self.encoder4(x1)

        x1 = self.middle_conv1(x1_)
        x2 = x1

        x1 = self.decoder1(x1, skip4)
        x1 = self.decoder2(x1, skip3)
        x1 = self.decoder3(x1, skip2)
        x1 = self.decoder4(x1, skip1)
        t = torch.sigmoid(self.final_conv1(x1))
        t = 0.9 * t + 0.1

        x2 = self.output1(x2)
        x2 = self.output2(x2)
        x2 = self.output3(x2)
        x2 = self.output4(x2)
        x2 = torch.sigmoid(self.final_conv2(x2))
        
        A = torch.mean(x2, dim=[2, 3], keepdim=True).expand(-1, -1, x.size(2), x.size(3))

        return A, t