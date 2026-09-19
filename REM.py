import torch
import torch.nn as nn
import utils
from RRM import RetinexReconstructionModule

class RetinexEnhancementModule(nn.Module):
    """
    REM: Retinex Enhancement Module
    从二次Retinex分解与CLAHE对比度增强部分开始的组合处理模块。
    """
    def __init__(self):
        super(RetinexEnhancementModule, self).__init__()
        self.rrm = RetinexReconstructionModule()

    def forward(self, x):
        min_channels, min_indices, mid_channels, mid_indices, max_channels, max_indices = utils.Split_channels(x)

        # 重构 Reflection 分量
        new_ref1s, comp_channels = self.rrm(max_channels)

        # 补偿 illumination 分量
        ref1s, ill1s = utils.Tensor_Retinex_decomposition(x, sigma=1)
        new_ill1s = ill1s + comp_channels

        # 合成补偿图像并进行二次 Retinex 分解与色彩调整
        new_imgs = utils.Tensor_Retinex_merge(new_ref1s, new_ill1s)
        new_imgs_clahe = utils.Tensor_CLAHE(new_imgs)
        new_ref2s, new_ill2s = utils.Tensor_Retinex_decomposition(new_imgs_clahe, sigma=100)
        wb_imgs = utils.Tensor_CLAHE(new_ref2s)

        return ref1s, new_ref1s, ill1s, new_ill1s, new_ref2s, new_ill2s, comp_channels, wb_imgs