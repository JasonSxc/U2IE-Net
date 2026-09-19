import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  # 用于可视化进度条

import utils
from REM import RetinexEnhancementModule
from MainModel import MainModel
import LossCaculate as loss_fn

# 1. 超参数与配置
lr = 0.0005
new_size = (256,256)
batch_size = 8   
epochs = 50
save_interval = 10  # 每隔多少个 Epoch 保存一次指定轮次权重

# 路径配置
output_dir = "./OpenData"
loss_save_path = os.path.join(output_dir, ".lossdata/training_loss.csv")
ckpt_dir = os.path.join(output_dir, "checkpoints")

u_path = "./underwater"
h_path = "./haze"
g_path = "./clear"

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# 2. 数据路径与加载 (加入可视化提示)
print(">>> 正在读取并构建数据集...")
paths = [u_path, h_path, g_path]

# 包装数据构建过程的可视化
# 修改 train.py 中的 DataLoader 实例化部分：
train_dataset = utils.build_data(paths, new_size=new_size, num_imgs=5000)

train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=4,        # 开启 4 个 CPU 子进程并行预读取图像（可根据机器 CPU 核心数适度增加）
    pin_memory=True,     # 开启锁页内存，加快 CPU 到 GPU (cuda:3) 的张量传输速度
    persistent_workers=True # 保持 Worker 进程不销毁，避免每个 Epoch 重新创建进程的开销
)
print(f">>> 数据集加载完成，共计 {len(train_dataset)} 样本，{len(train_loader)} 个 Batch。")

# 3. 初始化模型与优化器
rem = RetinexEnhancementModule()
model = MainModel(rem_module=rem).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# 4. 训练主逻辑
def train_model(model, train_loader, optimizer, num_epochs, save_dir, save_every):
    model.train()
    os.makedirs(save_dir, exist_ok=True)
    
    train_loss_history1, train_loss_history2, train_loss_history3, total_loss = [], [], [], []

    # 外层 Epoch 进度条
    epoch_bar = tqdm(range(num_epochs), desc="Training Progress", unit="epoch")
    
    for epoch in epoch_bar:
        running_loss1, running_loss2, running_loss3, running_total = 0.0, 0.0, 0.0, 0.0

        # 内层 Batch 进度条
        batch_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch + 1}/{num_epochs}", 
            leave=False, 
            unit="batch"
        )

        for U, H, G in batch_bar:
            U, H, G = U.to(device), H.to(device), G.to(device)

            # 打乱 U 的顺序
            U = U[torch.randperm(U.size(0))]
            
            # 前向计算
            t1, H2G, G2H, ref1s_h, new_ref1s_h, ill1s_h, new_ill1s_h = model(H=H, G=G)
            t2, new_ill1s_u, Uwb = model(U=U)

            # 损失函数计算
            l1 = (loss_fn.calculate_ssim_loss(ref1s_h, new_ref1s_h) + 
                  loss_fn.calculate_ssim_loss(H2G, G) + 
                  loss_fn.calculate_ssim_loss(G2H, H) + 
                  0.15 * loss_fn.calculate_ssim_loss(ill1s_h, new_ill1s_h))
            
            l2 = 0.005 * (loss_fn.caculate_variation_loss(Uwb) + 
                          loss_fn.caculate_variation_loss(t1) + 
                          loss_fn.caculate_variation_loss(t2))
            
            l3 = loss_fn.caculate_mid_difference_loss(new_ill1s_u) + loss_fn.caculate_max_difference_loss(new_ill1s_u)

            totalloss = l1 + l2 + l3
            
            # 反向传播
            optimizer.zero_grad()
            totalloss.backward() 
            optimizer.step()

            running_loss1 += l1.item()
            running_loss2 += l2.item()
            running_loss3 += l3.item()
            running_total += totalloss.item()

            # 实时更新 Batch 进度条的后缀 Loss 显示
            batch_bar.set_postfix({"Loss": f"{totalloss.item():.4f}"})

        num_batches = len(train_loader)
        epoch_loss1 = running_loss1 / num_batches
        epoch_loss2 = running_loss2 / num_batches
        epoch_loss3 = running_loss3 / num_batches
        epoch_total = running_total / num_batches

        train_loss_history1.append(epoch_loss1)
        train_loss_history2.append(epoch_loss2)
        train_loss_history3.append(epoch_loss3)
        total_loss.append(epoch_total)

        # 1) 保存最新权重 (latest.pth)
        latest_ckpt_path = os.path.join(save_dir, "u2ie_latest.pth")
        torch.save(model.state_dict(), latest_ckpt_path)

        # 2) 按指定轮次间隔保存指定 checkpoint
        current_epoch = epoch + 1
        if current_epoch % save_every == 0 or current_epoch == num_epochs:
            interval_ckpt_path = os.path.join(save_dir, f"u2ie_epoch_{current_epoch}.pth")
            torch.save(model.state_dict(), interval_ckpt_path)

        # 更新 Epoch 进度条后缀
        epoch_bar.set_postfix({
            "L1": f"{epoch_loss1:.3f}",
            "L2": f"{epoch_loss2:.3f}",
            "L3": f"{epoch_loss3:.3f}",
            "Total": f"{epoch_total:.3f}"
        })

    return train_loss_history1, train_loss_history2, train_loss_history3, total_loss

# 绘制训练曲线函数
def plot_loss_curves(l1, l2, l3, total, save_path):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.figure(figsize=(16, 5))
    epochs_range = range(1, len(total) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, l1, label='SSIM Loss (L1)', linewidth=2)
    plt.plot(epochs_range, l2, label='Variation Loss (L2)', linewidth=2)
    plt.plot(epochs_range, l3, label='Illumination Loss (L3)', linewidth=2)
    plt.title('Sub-Loss Curves', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, total, label='Total Loss', color='crimson', linewidth=2)
    plt.title('Total Loss Curve', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == '__main__':
    # 执行训练
    l1, l2, l3, total = train_model(
        model=model, 
        train_loader=train_loader, 
        optimizer=optimizer, 
        num_epochs=epochs,
        save_dir=ckpt_dir,
        save_every=save_interval
    )

    # 5. 保存 Loss 日志 CSV
    os.makedirs(os.path.dirname(loss_save_path), exist_ok=True)
    df = pd.DataFrame({
        'Epoch': list(range(1, epochs + 1)),
        'Loss1': l1,
        'Loss2': l2,
        'Loss3': l3,
        'Total_Loss': total
    })
    df.to_csv(loss_save_path, index=False)
    print(f"\n>>> 训练完成！Loss 数据已保存至: {loss_save_path}")

    # 6. 保存最终轮次模型 (保留您原先命名的权重文件路径)
    final_pth_path = os.path.join(ckpt_dir, "u2ie.pth")
    torch.save(model.state_dict(), final_pth_path)
    print(f">>> 最终权重已保存至: {final_pth_path}")

    # 7. 可视化绘制并保存 Losses 曲线图
    chart_save_path = os.path.join(output_dir, "lossdata/training_loss_curve.png")
    plot_loss_curves(l1, l2, l3, total, chart_save_path)
    print(f">>> 损失曲线图已生成并保存至: {chart_save_path}")