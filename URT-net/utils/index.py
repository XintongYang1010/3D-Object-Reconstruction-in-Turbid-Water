#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/22 16:47
# @Author  : jc Han
# @help    :




import numpy as np

def compute_min_norm_cpu(p1, p2):

    res = np.zeros(p1.shape[0], dtype=np.float32)


    for idx, p in enumerate(p1):

        dists = np.sqrt(np.sum((p2 - p) ** 2, axis=1))

        res[idx] = np.min(dists)

    return res

def cal_PCCD(p1_path, p2_path):

    P1 = np.genfromtxt(p1_path).astype(np.float32)
    P2 = np.genfromtxt(p2_path).astype(np.float32)


    res1 = compute_min_norm_cpu(P1, P2)
    res2 = compute_min_norm_cpu(P2, P1)


    result = 0.5 * (np.sum(res1) / len(P1) + np.sum(res2) / len(P2))
    print(result)
    return result


def cal_index_2D(path1,path2):

    import torch
    import lpips
    import torchvision.transforms as transforms
    import cv2
    from skimage.measure import compare_ssim

    # Load the two input images
    image1 = cv2.imread(path1)
    image2 = cv2.imread(path2)

    # Ensure images are of the same size
    image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

    # Calculate the PSNR between the two images
    psnr = cv2.PSNR(image1, image2)

    # Calculate the SSIM between the two images
    ssim = compare_ssim(image1, image2, multichannel=True)

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
    ])

    image1 = transform(image1).unsqueeze(0)  # Add batch dimension
    image2 = transform(image2).unsqueeze(0)  # Add batch dimension

    # Use GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the LPIPS metric
    loss_fn_vgg = lpips.LPIPS(net='vgg', verbose=False).to(device)

    # Move images to the device
    image1 = image1.to(device)
    image2 = image2.to(device)

    # Compute LPIPS distance
    LPIPS = loss_fn_vgg(image1, image2).item()

    return psnr,ssim,LPIPS

import glob
import os
def cal_all(path1,path2):

    redundancy = path1.split("/")[-1].split("_")
    pic_1_basename = redundancy[-2] + "_" + redundancy[-1] + "_1_" + os.path.basename(path2).split(".")[0] + "_moire.bmp"
    path1 = os.path.join(path1,pic_1_basename)
    psnr,ssim,LPIPS = cal_index_2D(path1,path2)

    return psnr,ssim,LPIPS


if __name__=="__main__":
    pic_1 = r""
    pic_2 = r""
    psnr,ssim,LPIPS=cal_index_2D(pic_1,pic_2)
    print(psnr,ssim,LPIPS)
