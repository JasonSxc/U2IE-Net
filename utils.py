import cv2
import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

# 标准化图像预处理 Pipeline (利用 torchvision GPU/CPU 高效算子)
def get_transforms(new_size):
    return T.Compose([
        T.Resize(new_size),
        T.ToTensor(),  # 自动转为 torch.float32 标量范围 [0, 1] 并且转成 (C, H, W)
    ])

class PairedImageData(Dataset):
    """
    训练集 Dataset (懒加载机制)：
    初始化只读取并配对文件路径/文件名，不占用系统内存；
    只有在 DataLoader 迭代时才由 CPU 多进程并行读取图像。
    """
    def __init__(self, paths, new_size, num_imgs=None):
        self.transform = get_transforms(new_size)
        
        min_file_count = min(len(os.listdir(path)) for path in paths)
        num_imgs = num_imgs if num_imgs and num_imgs <= min_file_count else min_file_count

        # 1. 提取文件名列表并排序
        u_files = sorted(os.listdir(paths[0]))
        h_files = sorted(os.listdir(paths[1]))
        g_files = sorted(os.listdir(paths[2]))

        # 2. 混洗打乱逻辑 (维持 H 与 G 配对，U 独立打乱)
        random.shuffle(u_files)
        h_g_paired = list(zip(h_files, g_files))
        random.shuffle(h_g_paired)
        h_files, g_files = zip(*h_g_paired)

        # 3. 拦截截取 num_imgs 长度的文件绝对路径
        self.u_paths = [os.path.join(paths[0], f) for f in u_files[:num_imgs]]
        self.h_paths = [os.path.join(paths[1], f) for f in h_files[:num_imgs]]
        self.g_paths = [os.path.join(paths[2], f) for f in g_files[:num_imgs]]

    def __len__(self):
        return len(self.u_paths)

    def __getitem__(self, idx):
        # 实时按需读取单张图片
        u_img = Image.open(self.u_paths[idx]).convert('RGB')
        h_img = Image.open(self.h_paths[idx]).convert('RGB')
        g_img = Image.open(self.g_paths[idx]).convert('RGB')

        return self.transform(u_img), self.transform(h_img), self.transform(g_img)


class TestImageData(Dataset):
    """测试集 Dataset (无 H-G 配对要求，纯独立打乱)"""
    def __init__(self, paths, new_size, num_imgs=None):
        self.transform = get_transforms(new_size)
        
        min_file_count = min(len(os.listdir(path)) for path in paths)
        num_imgs = num_imgs if num_imgs and num_imgs <= min_file_count else min_file_count

        all_file_paths = []
        for path in paths:
            filenames = sorted(os.listdir(path))
            random.shuffle(filenames)
            all_file_paths.append([os.path.join(path, f) for f in filenames[:num_imgs]])

        self.img_paths_list = all_file_paths

    def __len__(self):
        return len(self.img_paths_list[0])

    def __getitem__(self, idx):
        imgs = [Image.open(paths[idx]).convert('RGB') for paths in self.img_paths_list]
        return tuple(self.transform(img) for img in imgs)


# 保留对外的统一构建接口
def build_data(paths, new_size, num_imgs):
    return PairedImageData(paths, new_size, num_imgs)

def build_test_data(paths, new_size, num_imgs):
    return TestImageData(paths, new_size, num_imgs)

def freezing_parameters(model):
    for param in model.base_model.parameters():
        param.requires_grad = False
    for param in model.DownstreamModel.parameters():
        param.requires_grad = True
    return model

def GaussianBlur(img, sigma=10):
    return cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma)

def batch_GussianBlur(batch_tensor, sigma=10):
    batch_np = (batch_tensor.permute(0, 2, 3, 1).detach().cpu().numpy() * 255.0).astype(np.float32)
    for i in range(batch_np.shape[0]):
        batch_np[i] = GaussianBlur(batch_np[i], sigma=sigma)
    return torch.from_numpy(batch_np / 255.0).permute(0, 3, 1, 2).to(batch_tensor.device, dtype=batch_tensor.dtype)

def RGB_CLAHE(image, tile_grid_size=(1, 1), cl=2.0):
    channels = cv2.split(np.uint8(image))
    clahe = cv2.createCLAHE(clipLimit=cl, tileGridSize=tile_grid_size)
    eq_channels = [cv2.normalize(clahe.apply(ch), None, 0, 255, cv2.NORM_MINMAX) for ch in channels]
    return cv2.merge(eq_channels)

def Retinex_decomposition(img, sigma=200):
    img_f = img.astype(np.float32)
    log_img = np.log(img_f + 1e-4)
    illumination = cv2.GaussianBlur(img_f, (0, 0), sigma)
    log_blur = np.log(illumination + 1e-4)
    
    reflection = np.clip(np.exp(log_img - log_blur), 0, 5.6)
    reflection = cv2.normalize(reflection, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    illumination = cv2.normalize(illumination, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return reflection, illumination

def Retinex_merge(ref, ill):
    img = ref.astype(np.float32) * ill.astype(np.float32)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def Tensor_Retinex_decomposition(batch_imgs, sigma=100):
    batch_np = (batch_imgs.permute(0, 2, 3, 1).detach().cpu().numpy() * 255.0)
    refs_np, ills_np = np.empty_like(batch_np), np.empty_like(batch_np)
    
    for i in range(batch_np.shape[0]):
        ref, ill = Retinex_decomposition(batch_np[i], sigma=sigma)
        refs_np[i], ills_np[i] = ref / 255.0, ill / 255.0
        
    refs = torch.from_numpy(refs_np).permute(0, 3, 1, 2).to(batch_imgs.device, dtype=batch_imgs.dtype)
    ills = torch.from_numpy(ills_np).permute(0, 3, 1, 2).to(batch_imgs.device, dtype=batch_imgs.dtype)
    return refs, ills

def Tensor_Retinex_merge(batch_refs, batch_ills):
    refs_np = batch_refs.permute(0, 2, 3, 1).detach().cpu().numpy() * 255.0
    ills_np = batch_ills.permute(0, 2, 3, 1).detach().cpu().numpy() * 255.0
    imgs_np = np.empty_like(refs_np)
    
    for i in range(refs_np.shape[0]):
        imgs_np[i] = Retinex_merge(refs_np[i], ills_np[i]) / 255.0
        
    return torch.from_numpy(imgs_np).permute(0, 3, 1, 2).to(batch_ills.device, dtype=batch_ills.dtype)

def Mean_caculate(img):
    return any(np.mean(ch) < 128 for ch in cv2.split(img))

def Tensor_CLAHE(batch_imgs, tile_grid_size=(1, 1), cl=1.0):
    batch_np = batch_imgs.permute(0, 2, 3, 1).detach().cpu().numpy() * 255.0
    res_np = np.empty_like(batch_np)
    
    for i in range(batch_np.shape[0]):
        img = batch_np[i]
        limit = cl if Mean_caculate(img) else 2.0
        res_np[i] = RGB_CLAHE(img, tile_grid_size=tile_grid_size, cl=limit) / 255.0
        
    return torch.from_numpy(res_np).permute(0, 3, 1, 2).to(batch_imgs.device, dtype=batch_imgs.dtype)

def Split_channels(x):
    # 高效向量化通道分拆
    B, C, H, W = x.size()
    channel_means = x.mean(dim=[2, 3])
    sorted_indices = torch.argsort(channel_means, dim=1)
    
    min_idx = sorted_indices[:, 0].view(B, 1, 1, 1).expand(-1, 1, H, W)
    mid_idx = sorted_indices[:, 1].view(B, 1, 1, 1).expand(-1, 1, H, W)
    max_idx = sorted_indices[:, 2].view(B, 1, 1, 1).expand(-1, 1, H, W)
    
    min_channels = torch.gather(x, 1, min_idx)
    mid_channels = torch.gather(x, 1, mid_idx)
    max_channels = torch.gather(x, 1, max_idx)
    
    return min_channels, sorted_indices[:, 0], mid_channels, sorted_indices[:, 1], max_channels, sorted_indices[:, 2]