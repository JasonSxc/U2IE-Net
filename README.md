# U2IE-Net: An Unpaired Underwater Optical Image Enhancement Network Based on Multi-Physical Model Constraints

[![Journal](https://img.shields.io/badge/Journal-Neurocomputing-blue.svg)](https://doi.org/10.1016/j.neucom.2026.134993)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[English](#english) | [中文说明](#中文说明)

---

<a name="english"></a>
## English

### 📌 Overview
This is the official repository for **U2IE Net** (An unpaired underwater optical images enhancement network based on multi-physical model constraints), published in *Neurocomputing*. 

Underwater images often suffer from severe degradation caused by light absorption, scattering, and suspended particles. To overcome the reliance on strictly paired underwater datasets, **U2IE Net** utilizes a novel cross-domain paired supervision strategy with multi-physical degradation model constraints, delivering optimal image restoration, structural preservation, and computational efficiency.

---

### 📥 Pre-trained Weights Download
You can download the pre-trained checkpoint file (`u2ie_latest.pth`) from the following cloud storage links:

- **Baidu Netdisk (百度网盘):** 
  - Link: [Download via Baidu Netdisk](https://pan.baidu.com/s/1M9zsRGpNaj3oNugQiWIWFA?pwd=733i password: 733i)
- **Quark Netdisk (夸克网盘):** 
  - Link: [Download via Quark Netdisk](https://pan.quark.cn/s/767d1f73b67a?pwd=Kggn password: Kggn)

Place the downloaded weight file (e.g., `u2ie_latest.pth`) inside a directory such as `./checkpoints/`.

---

### 🛠️ Quick Start & Inference
The inference framework is designed to run efficiently via batch processing (`test.py`).

#### 1. Environment Requirements
- Python 3.8+
- PyTorch 1.10+
- torchvision
- Pillow
- tqdm

Install dependencies using:
```bash
pip install torch torchvision pillow tqdm
```

#### 2. Project Directory Structure
Ensure your project directory is organized as follows:
```text
U2IE-Net/
│
├── REM.py                       # Retinex Enhancement Module
├── MainModel.py                 # Main Model Architecture
├── test.py                      # Inference Script
└── checkpoints/
    └── u2ie_latest.pth          # Pre-trained Weights
```

#### 3. Running Inference
To run the model on your custom dataset:
1. Open `test.py` and modify the path parameters in the `if __name__ == '__main__':` section:
   - `INPUT_DIR`: Path to your raw underwater images.
   - `OUTPUT_DIR`: Directory where enhanced images will be saved.
   - `CHECKPOINT_PATH`: Path to your downloaded `.pth` weight file.
   - `IMAGE_SIZE`: Target processing resolution (default: `(256, 256)`).
   - `BATCH_SIZE`: Batch size for batch processing (default: `8`).
   - `DEVICE`: Target CUDA device (e.g., `'cuda:0'`, `'cuda:2'`, or `'cpu'`).
2. Run the inference script:
   ```bash
   python test.py
   ```

*Supported input formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`.*

---

### 📖 Citation
If you find our work or code useful in your research, please consider citing our paper:

```bibtex
@article{SHUANG2027134993,
title = {An unpaired underwater optical images enhancement network based on multi-physical model constraints},
journal = {Neurocomputing},
volume = {707},
pages = {134993},
year = {2027},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2026.134993},
url = {https://www.sciencedirect.com/science/article/pii/S092523122602391X},
author = {Xuecheng Shuang and Yan Huang and Jianan Qiao and Hao Feng and Dayu Jia},
keywords = {Underwater image enhancement, Unpaired learning, Multi-physical model constraints, Retinex model, Image formation model}
}
```

---

<a name="中文说明"></a>
## 中文说明

### 📌 项目简介
本仓库是发表于 *Neurocomputing* 期刊的论文 **U2IE Net**（基于多物理模型约束的无监督水下光学图像增强网络）的官方实现。

针对水下图像因吸收和散射导致的严重色偏与结构模糊问题，**U2IE Net** 提出了跨领域对齐监督策略与多物理退化模型约束，摆脱了对真实水下成对数据集的依赖，在保持高效率计算的同时实现了优秀的去雾与水下复原效果。

---

### 📥 预训练权重下载
您可以通过以下网盘链接下载预训练权重文件（`u2ie_latest.pth`）：

- **百度网盘:**
  - 链接: https://pan.baidu.com/s/1M9zsRGpNaj3oNugQiWIWFA?pwd=733i 提取码: 733i
- **夸克网盘:**
  - 链接：https://pan.quark.cn/s/767d1f73b67a?pwd=Kggn 提取码：Kggn

下载后建议将权重文件保存至项目下的 `./checkpoints/` 路径。

---

### 🛠️ 快速上手与推理指南

本项目提供了极速批处理推理脚本（`test.py`），可快速对自定义水下图像文件夹进行增强。

#### 1. 环境准备
- Python 3.8+
- PyTorch 1.10+
- torchvision
- Pillow
- tqdm

安装依赖库：
```bash
pip install torch torchvision pillow tqdm
```

#### 2. 项目目录结构
请确保项目包含以下核心文件：
```text
U2IE-Net/
│
├── REM.py                       # Retinex 增强模块
├── MainModel.py                 # 主模型结构定义
├── test.py                      # 推理运行脚本
└── checkpoints/
    └── u2ie_latest.pth          # 下载的模型权重文件
```

#### 3. 执行推理
1. 打开 `test.py` 脚本，在 `if __name__ == '__main__':` 入口处修改参数：
   - `INPUT_DIR`: 待增强的原始水下图像文件夹路径。
   - `OUTPUT_DIR`: 增强结果保存的目标文件夹路径。
   - `CHECKPOINT_PATH`: 下载好的 `.pth` 权重文件路径。
   - `IMAGE_SIZE`: 推理分辨率（默认：`(256, 256)`）。
   - `BATCH_SIZE`: 批处理大小（默认：`8`）。
   - `DEVICE`: 推理计算设备（例如 `'cuda:0'`、`'cuda:2'` 或 `'cpu'`）。
2. 在终端运行推理命令：
   ```bash
   python test.py
   ```

*支持的输入图像格式：`.png`、`.jpg`、`.jpeg`、`.bmp`、`.tif`、`.tiff`。*

---

### 📖 论文引用
如果本研究或代码对您的科研工作有所帮助，请考虑引用我们的论文：

```bibtex
@article{SHUANG2027134993,
title = {An unpaired underwater optical images enhancement network based on multi-physical model constraints},
journal = {Neurocomputing},
volume = {707},
pages = {134993},
year = {2027},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2026.134993},
url = {https://www.sciencedirect.com/science/article/pii/S092523122602391X},
author = {Xuecheng Shuang and Yan Huang and Jianan Qiao and Hao Feng and Dayu Jia},
keywords = {Underwater image enhancement, Unpaired learning, Multi-physical model constraints, Retinex model, Image formation model}
}
```
