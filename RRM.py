import torch
import torch.nn as nn
import utils
from Blocks import Encoder, ConvBlock, Decoder, Decoder_Bilinear

class RetinexReconstructionModule(nn.Module):
    """
    RRM: Retinex Reconstruction Module
    用于从图像单通道提取反射分量与光源分量补偿信息。
    """
    def __init__(self):
        super(RetinexReconstructionModule, self).__init__()
        self.encoder1 = Encoder(1, 64, use_cbam=True)
        self.encoder2 = Encoder(64, 128, use_cbam=True)
        self.encoder3 = Encoder(128, 256, use_cbam=True)

        self.middle_conv1 = ConvBlock(256, 512, use_cbam=True)

        self.decoder1_1 = Decoder(512, 256, use_cbam=True)     
        self.decoder2_1 = Decoder(256, 128, use_cbam=True)
        self.decoder3_1 = Decoder_Bilinear(128, 64, use_cbam=True)
        self.final_conv1 = nn.Conv2d(64, 3, kernel_size=1)

        self.middle_conv2 = ConvBlock(256, 512, use_cbam=True)
        self.decoder1_2 = Decoder(512, 256, use_cbam=True)     
        self.decoder2_2 = Decoder(256, 128, use_cbam=True)
        self.decoder3_2 = Decoder_Bilinear(128, 64, use_cbam=True)
        self.final_conv2 = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        skip1, x = self.encoder1(x)
        skip2, x = self.encoder2(x)
        skip3, x = self.encoder3(x)

        x1 = self.middle_conv1(x)
        x2 = x1

        x1 = self.decoder1_1(x1, skip3)
        x1 = self.decoder2_1(x1, skip2)
        x1 = self.decoder3_1(x1, skip1)
        new_ref = torch.sigmoid(self.final_conv1(x1))   

        x2 = self.decoder1_2(x2, skip3)
        x2 = self.decoder2_2(x2, skip2)
        x2 = self.decoder3_2(x2, skip1)
        ill_com = utils.batch_GussianBlur((0.5 * torch.sigmoid(self.final_conv2(x2))) * 255.0, sigma=100) / 255.0

        return new_ref, ill_com