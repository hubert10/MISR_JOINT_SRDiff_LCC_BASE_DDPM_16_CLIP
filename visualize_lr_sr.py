import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_multiband_image(path):
    """Reads a 4-channel image and returns only the RGB channels"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # shape: H x W x 4
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    if img.shape[2] != 4:
        raise ValueError(f"Image does not have 4 channels: {path}")
    img_rgb = img[:, :, :3]  # R, G, B (drop NIR)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
    return img_rgb


def plot_time_series(lr_folder, sr_folder):
    lr_images = sorted(
        [
            os.path.join(lr_folder, f)
            for f in os.listdir(lr_folder)
            if f.endswith((".png", ".tif", ".jpg"))
        ]
    )
    sr_images = sorted(
        [
            os.path.join(sr_folder, f)
            for f in os.listdir(sr_folder)
            if f.endswith((".png", ".tif", ".jpg"))
        ]
    )

    n = min(len(lr_images), len(sr_images))
    plt.figure(figsize=(12, 4 * n))

    for i in range(n):
        lr_img = read_multiband_image(lr_images[i])
        sr_img = read_multiband_image(sr_images[i])

        # Plot LR
        plt.subplot(n, 2, 2 * i + 1)
        plt.imshow(lr_img)
        plt.title(f"LR Image {i+1}")
        plt.axis("off")

        # Plot SR
        plt.subplot(n, 2, 2 * i + 2)
        plt.imshow(sr_img)
        plt.title(f"SR Image {i+1}")
        plt.axis("off")

    plt.show()


lr_folder = "D:\\kanyamahanga\\Datasets\\MISR_S2_Aer_LCC_x10_JOINT_SRDiff_SEG_SegFormer_HR_ConvFormer_SR_NIR_OPT_DATA_AUG_LPIPS\\LR\\D022_2021\\Z1_AA\\sen\\"
sr_folder = "D:\\kanyamahanga\\Datasets\\MISR_S2_Aer_LCC_x10_JOINT_SRDiff_SEG_SegFormer_HR_ConvFormer_SR_NIR_OPT_DATA_AUG_LPIPS\\SR\\D022_2021\\Z1_AA\\sen\\"

# lr_folder = "D:\\kanyamahanga\\Datasets\\MISR_S2_Aer_LCC_x10_JOINT_SRDiff_SEG_SegFormer_HR_ConvFormer_SR_NIR_OPT_DATA_AUG_LPIPS\\LR\\D015_2020\\Z16_UA\\sen\\"
# sr_folder = "D:\\kanyamahanga\\Datasets\\MISR_S2_Aer_LCC_x10_JOINT_SRDiff_SEG_SegFormer_HR_ConvFormer_SR_NIR_OPT_DATA_AUG_LPIPS\\SR\\D015_2020\\Z16_UA\\sen\\"


plot_time_series(lr_folder, sr_folder)


#!/usr/bin/env python
# coding: utf-8

import os
import random
import matplotlib.pyplot as plt
from glob import glob
import cv2
import re
import torch
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import rasterio.plot as plot
import torchvision.transforms as T
import rasterio
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
import rasterio
from data.data_display import convert_to_color, lut_colors


def load_hr_raster(img_id, rel_path, img_root):
    """
    Load a raster mask from a given image ID and relative path.

    Parameters:
    - img_id: str, e.g. "036385"
    - rel_path: str, e.g. "D055_2018/Z3_UF"
    - msk_root: str, base folder where masks are stored

    Returns:
    - mask: numpy array of the mask
    - profile: rasterio profile (metadata)
    - path: full path to the mask
    """
    img_filename = f"IMG_{img_id}.tif"
    img_path = os.path.join(img_root, rel_path, "img", img_filename)

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"IMG file not found: {img_path}")

    with rasterio.open(img_path, "r") as f:
        img = f.read([1, 2, 3]).swapaxes(0, 2).swapaxes(0, 1)
    print("img:", img.shape)
    return img


def load_mask_raster(img_id, rel_path, msk_root):
    """
    Load a raster mask from a given image ID and relative path.

    Parameters:
    - img_id: str, e.g. "036385"
    - rel_path: str, e.g. "D055_2018/Z3_UF"
    - msk_root: str, base folder where masks are stored

    Returns:
    - mask: numpy array of the mask
    - profile: rasterio profile (metadata)
    - path: full path to the mask
    """
    msk_filename = f"MSK_{img_id}.tif"
    msk_path = os.path.join(msk_root, rel_path, "msk", msk_filename)

    if not os.path.exists(msk_path):
        raise FileNotFoundError(f"Mask file not found: {msk_path}")

    with rasterio.open(msk_path, "r") as f:
        mk = f.read([1])
        print("mk:", mk.shape)

    mask = torch.as_tensor(mk, dtype=torch.int32)
    print("mask:", mask.shape)
    return mask


def load_pr_mask_raster(img_id, rel_path, msk_root):
    """
    Load a raster mask from a given image ID and relative path.

    Parameters:
    - img_id: str, e.g. "036385"
    - rel_path: str, e.g. "D055_2018/Z3_UF"
    - msk_root: str, base folder where masks are stored

    Returns:
    - mask: numpy array of the mask
    - profile: rasterio profile (metadata)
    - path: full path to the mask
    """
    msk_filename = f"PRED_{img_id}.tif"
    msk_path = os.path.join(msk_root, rel_path, "PR", msk_filename)
    msk_path = os.path.join(msk_root, msk_filename)

    if not os.path.exists(msk_path):
        raise FileNotFoundError(f"Mask file not found: {msk_path}")

    with rasterio.open(msk_path, "r") as f:
        mk = f.read([1])
        print("mk:", mk.shape)

    mask = torch.as_tensor(mk, dtype=torch.int32)
    print("mask:", mask.shape)
    return mask


def downsample_majority_vote_with_crop(
    labels, original_size=512, cropped_size=500, output_size=10
):
    """
    Downsamples multi-class label maps using majority vote after cropping.

    Args:
        labels (torch.Tensor): Input label maps of shape [N, 512, 512].
        original_size (int): Original spatial size (assumed square). Default is 512.
        cropped_size (int): Desired spatial size after cropping (assumed square). Default is 500.
        output_size (int): Desired output spatial size (assumed square). Default is 10.

    Returns:
        torch.Tensor: Downsampled label maps of shape [N, 10, 10].
    """
    N, H, W = labels.shape
    assert (
        H == original_size and W == original_size
    ), f"Input label maps must be of shape [N, {original_size}, {original_size}]"
    assert (
        cropped_size % output_size == 0
    ), f"cropped_size must be divisible by output_size. Got cropped_size={cropped_size}, output_size={output_size}"

    # Step 1: Crop the label maps to [N, 500, 500]
    # Assuming center crop: remove 6 pixels from each side
    crop_margin = (original_size - cropped_size) // 2  # 6 pixels
    labels_cropped = labels[
        :,
        crop_margin : crop_margin + cropped_size,
        crop_margin : crop_margin + cropped_size,
    ]

    # Step 2: Reshape to [N, output_size, block_size, output_size, block_size]
    block_size = cropped_size // output_size  # 50
    labels_reshaped = labels_cropped.view(
        N, output_size, block_size, output_size, block_size
    )

    # Step 3: Permute to [N, output_size, output_size, block_size, block_size]
    labels_permuted = labels_reshaped.permute(0, 1, 3, 2, 4)

    # Step 4: Flatten the block pixels to [N, output_size, output_size, block_size * block_size]
    labels_flat = labels_permuted.reshape(
        N, output_size, output_size, block_size * block_size
    )

    # Step 5: Compute mode along the last dimension (majority vote)
    mode, _ = torch.mode(labels_flat, dim=-1)

    return mode  # [N, 10, 10]


def plot_random_hr_lr_sr(hr_root, lr_root, sr_root, pr_root):
    # Step 1: Randomly select an HR image
    hr_images = glob(os.path.join(hr_root, "**", "img", "*.png"), recursive=True)
    if not hr_images:
        print("No HR images found.")
        return

    hr_img_path = random.choice(hr_images)
    print(f"Selected HR image: {hr_img_path}")

    # Step 2: Extract ID and corresponding LR/SR folders
    base_name = os.path.basename(hr_img_path)  # e.g., IMG_077413.png
    img_id = base_name.split("_")[-1].split(".")[0]  # '077413'

    # Get the relative path after HR root (e.g., D015_2020/Z1_AA)
    rel_path = os.path.relpath(os.path.dirname(os.path.dirname(hr_img_path)), hr_root)

    lr_folder = os.path.join(lr_root, rel_path, "sen", img_id)
    sr_folder = os.path.join(sr_root, rel_path, "sen", img_id)

    # Step 3: Load images and masks
    hr_img = load_hr_raster(img_id, rel_path, img_root_folder)
    msk = load_mask_raster(img_id, rel_path, msk_root_folder)
    pr_msk = load_pr_mask_raster(img_id, rel_path, pr_root)

    print("HR image:", hr_img.shape)
    print("HR mask:", msk.shape)
    print("PR mask:", pr_msk.shape)

    msk_color = convert_to_color(msk[0], palette=lut_colors)
    pr_msk_color = convert_to_color(pr_msk[0], palette=lut_colors)

    # Load LR and SR images
    lr_imgs = sorted(glob(os.path.join(lr_folder, "*.png")))
    sr_imgs = sorted(glob(os.path.join(sr_folder, "*.png")))
    lr_images = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in lr_imgs]
    sr_images = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in sr_imgs]

    # Step 4: Prepare one-row plot
    all_images = [hr_img, msk_color, pr_msk_color]
    titles = ["HR Image", "GT Mask", "Pred Mask"]

    fig_cols = len(all_images)
    fig, axs = plt.subplots(1, fig_cols, figsize=(4 * fig_cols, 6))

    for i, (img, title) in enumerate(zip(all_images, titles)):
        axs[i].imshow(img)
        axs[i].set_title(title)
        axs[i].axis("off")

    plt.tight_layout()
    plt.show()


root_folder = "D:\\kanyamahanga\\Datasets\\MISR_JOINT_SRDiff_LCC_BASE_DDPM_16_CLIP_MaxViT_HR5_ConvFormer_SR4_CAF_FOCAL_ALL\\"
img_root_folder = "D:\\kanyamahanga\\Datasets\\FLAIR\\flair_aerial_test\\"
msk_root_folder = "D:\\kanyamahanga\\Datasets\\FLAIR\\flair_labels_test\\"
plot_random_hr_lr_sr(
    hr_root=root_folder + "HR",
    lr_root=root_folder + "LR",
    sr_root=root_folder + "SR",
    pr_root=root_folder + "PR",
)


# 	IMG_078678
# Path	D:\kanyamahanga\Datasets\FLAIR\flair_aerial_test\D015_2020\Z13_FN\img\IMG_078678.tif
# Sidecar file	IMG_078678.tif.aux.xml
# Total size	1.25 MB
# Last modified	Wednesday, May 17, 2023 10:10:44 PM (IMG_078678.tif.aux.xml)
# Provider	gdal
