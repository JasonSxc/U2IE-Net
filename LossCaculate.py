import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

def calculate_ssim_loss(image_batch1, image_batch2, window_size=11):
    ssim_loss_fn = kornia.losses.SSIMLoss(window_size, reduction='mean')
    return ssim_loss_fn(image_batch1, image_batch2)

def calculate_class_loss(image_batch, Classifier, label):
    batch_size = image_batch.size(0)
    outputs = Classifier(image_batch)
    labels = torch.full((batch_size,), fill_value=label, dtype=torch.long, device=image_batch.device)
    return F.cross_entropy(outputs, labels)

def caculate_variation_loss(x):
    return torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])) + \
           torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))

def caculate_ColorConstancy_loss(output, target):
    output_mean, output_var = torch.mean(output, dim=[0, 2, 3]), torch.var(output, dim=[0, 2, 3])
    target_mean, target_var = torch.mean(target, dim=[0, 2, 3]), torch.var(target, dim=[0, 2, 3])
    return F.mse_loss(output_mean, target_mean) + F.mse_loss(output_var, target_var)

def caculate_mse_loss(output, target):
    return F.mse_loss(output, target)

def caculate_grayscale_constraint_loss(output):
    R, G, B = output[:, 0, :, :], output[:, 1, :, :], output[:, 2, :, :]
    return torch.mean(torch.abs(R - G)) + torch.mean(torch.abs(G - B)) + torch.mean(torch.abs(R - B))

def compute_gradients(image):
    device = image.device
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device).view(1, 1, 3, 3).repeat(image.size(1), 1, 1, 1)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device).view(1, 1, 3, 3).repeat(image.size(1), 1, 1, 1)
    grad_x = F.conv2d(image, sobel_x, padding=1, groups=image.size(1))
    grad_y = F.conv2d(image, sobel_y, padding=1, groups=image.size(1))
    return grad_x, grad_y

def caculate_gradient_difference_loss(prediction, target):
    grad_pred_x, grad_pred_y = compute_gradients(prediction)
    grad_target_x, grad_target_y = compute_gradients(target)
    return torch.mean(torch.abs(grad_pred_x - grad_target_x)) + torch.mean(torch.abs(grad_pred_y - grad_target_y))

def caculate_brightest_loss(background_light, reference_image, top_percent=0.005):
    B, C, H, W = reference_image.shape
    grayscale = torch.mean(reference_image, dim=1).view(B, -1)
    top_k = max(1, int(top_percent * grayscale.shape[1]))
    _, top_indices = torch.topk(grayscale, top_k, dim=-1, largest=True)
    
    # 向量化索引提速
    top_indices_expanded = top_indices.unsqueeze(1).expand(-1, C, -1)
    ref_flat = reference_image.view(B, C, -1)
    selected_values = torch.gather(ref_flat, 2, top_indices_expanded)
    
    target_background_light = selected_values.mean(dim=-1).view(B, C, 1, 1)
    return caculate_mse_loss(background_light, target_background_light)

def caculate_max_difference_loss(images):
    channels_means = images.mean(dim=[2, 3])
    sorted_means, _ = torch.sort(channels_means, dim=1)
    return caculate_mse_loss(sorted_means[:, 0], sorted_means[:, 2])

def caculate_mid_difference_loss(images):
    channels_means = images.mean(dim=[2, 3])
    sorted_means, _ = torch.sort(channels_means, dim=1)
    return caculate_mse_loss(sorted_means[:, 0], sorted_means[:, 1])