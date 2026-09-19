import torch
import torch.nn as nn
from REM import RetinexEnhancementModule
from PDM import PhysicalDehazingModule

class MainModel(nn.Module):
    def __init__(self, rem_module: nn.Module = None):
        super(MainModel, self).__init__()
        self.dual_branch_model = PhysicalDehazingModule()
        self.RR = rem_module if rem_module is not None else RetinexEnhancementModule()

    def forward(self, H=None, U=None, G=None):
        if H is not None:
            ref1s_h, new_ref1s_h, ill1s_h, new_ill1s_h, new_ref2s_h, new_ill2s_h, new_imgs_h, Hwb = self.RR(H)

            A1, t1 = self.dual_branch_model(H)
            H2G = (H - A1 * (1 - t1)) / t1

            if G is not None:
                G2H = G * t1 + A1 * (1 - t1)
                if self.training:
                    return t1, H2G, G2H, ref1s_h, new_ref1s_h, ill1s_h, new_ill1s_h
            else:
                return A1, t1, H2G, ref1s_h, new_ref1s_h, ill1s_h, new_ill1s_h, new_ref2s_h, new_ill2s_h, new_imgs_h, Hwb
                
        elif U is not None:
            ref1s_u, new_ref1s_u, ill1s_u, new_ill1s_u, new_ref2s_u, new_ill2s_u, comp_channels, Uwb = self.RR(U)

            A2, t2 = self.dual_branch_model(Uwb)
            Uwb2G = (Uwb - A2 * (1 - t2)) / t2

            if self.training:
                return t2, new_ill1s_u, Uwb
            else:
                return A2, t2, Uwb2G, ref1s_u, new_ref1s_u, ill1s_u, new_ill1s_u, new_ref2s_u, new_ill2s_u, comp_channels, Uwb
        else:
            raise ValueError("At least one of H or U must be provided.")