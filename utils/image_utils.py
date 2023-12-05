#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from skimage.measure import compare_ssim as ssim

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))



def ssim(tensor1, tensor2):
    """
    Calculate the Structural Similarity (SSIM) between two tensors.

    Args:
    tensor1 (torch.Tensor): The first image tensor.
    tensor2 (torch.Tensor): The second image tensor.

    Returns:
    float: The SSIM index.
    """
    # Convert tensors to numpy arrays
    # Assumes tensors are on CPU and have shape (C, H, W) and are in range [0, 1]
    img1 = tensor1.permute(1, 2, 0).cpu().numpy()
    img2 = tensor2.permute(1, 2, 0).cpu().numpy()

    # Convert to grayscale if they are RGB
    if img1.shape[2] == 3:
        img1 = np.dot(img1[..., :3], [0.2989, 0.5870, 0.1140])
        img2 = np.dot(img2[..., :3], [0.2989, 0.5870, 0.1140])

    # Calculate SSIM
    return ssim(img1, img2, data_range=img1.max() - img1.min())
