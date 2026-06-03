#!/usr/bin/env python
# coding: utf-8

import os
import random
import matplotlib.pyplot as plt
from glob import glob
import torch
import random
import rasterio
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


def load_pr_mask_raster(img_id, msk_root):
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
    msk_path = os.path.join(msk_root, msk_filename)

    if not os.path.exists(msk_path):
        raise FileNotFoundError(f"Mask file not found: {msk_path}")

    with rasterio.open(msk_path, "r") as f:
        mk = f.read()
    mask = torch.as_tensor(mk, dtype=torch.int32)
    return mask


def plot_random_hr_lr_sr(hr_root, pr_root):
    # Step 1: Randomly select an HR image
    hr_images = glob(os.path.join(hr_root, "**", "img", "*.tif"), recursive=True)
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

    # Step 3: Load images and masks
    hr_img = load_hr_raster(img_id, rel_path, img_root_folder)
    msk = load_mask_raster(img_id, rel_path, msk_root_folder)
    pr_msk = load_pr_mask_raster(img_id, pr_root)
    pr_msk = pr_msk + 1

    # Convert to long and squeeze out channel dimension
    msk = msk.squeeze().long()
    pr_msk = pr_msk.squeeze().long()

    # Ensure they have the same unique class IDs
    print("GT unique:", torch.unique(msk))
    print("PR unique:", torch.unique(pr_msk))

    # Optional: Clip predicted IDs to valid range
    max_class = max(lut_colors.keys())
    pr_msk = torch.clamp(pr_msk, 0, max_class)

    # Now convert to color using the SAME palette
    msk_color = convert_to_color(msk, palette=lut_colors)
    pr_msk_color = convert_to_color(pr_msk, palette=lut_colors)

    max_class = max(lut_colors.keys())
    pr_msk = torch.clamp(pr_msk, 0, max_class)

    print(
        "GT dtype:",
        msk.dtype,
        "min:",
        msk.min().item(),
        "max:",
        msk.max().item(),
        "unique:",
        torch.unique(msk),
    )
    print(
        "PR dtype:",
        pr_msk.dtype,
        "min:",
        pr_msk.min().item(),
        "max:",
        pr_msk.max().item(),
        "unique:",
        torch.unique(pr_msk),
    )
    print("GT shape:", msk.shape, "PR shape:", pr_msk.shape)

    print("HR image:", hr_img.shape)
    print("HR mask:", msk.shape)
    print("PR mask:", pr_msk.shape)

    msk_color = convert_to_color(msk, palette=lut_colors)
    pr_msk_color = convert_to_color(pr_msk, palette=lut_colors)

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


root_folder = "D:\\kanyamahanga\\Bigwork\\MISR_JOINT_SRDiff_LCC_BASE_DDPM_16_CLIP\\"
img_root_folder = "D:\\kanyamahanga\\Datasets\\FLAIR_FN\\flair_aerial_test\\"
msk_root_folder = "D:\\kanyamahanga\\Datasets\\FLAIR_FN\\flair_labels_test\\"
plot_random_hr_lr_sr(hr_root=img_root_folder, pr_root=root_folder + "PR")


# 	IMG_078678
# Path	D:\kanyamahanga\Datasets\FLAIR\flair_aerial_test\D015_2020\Z13_FN\img\IMG_078678.tif
# Sidecar file	IMG_078678.tif.aux.xml
# Total size	1.25 MB
# Last modified	Wednesday, May 17, 2023 10:10:44 PM (IMG_078678.tif.aux.xml)
# Provider	gdal


# IMG_6349139.tif
# IMG_079039.tif
# IMG_077627.tif
# IMG_079048.tif
# IMG_079006.tif


# IMG_078689.tif
