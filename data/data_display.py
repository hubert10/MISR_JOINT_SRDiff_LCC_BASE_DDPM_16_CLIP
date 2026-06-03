#!/usr/bin/env python
# coding: utf-8


## Imports
import os
import re
import random
from pathlib import Path
import numpy as np
import matplotlib
from matplotlib.colors import hex2color
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import ImageGrid
import rasterio
import rasterio.plot as plot
import torch
import torchvision.transforms as T
import datetime
import datetime
import torch
import rasterio
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image


lut_colors = {
    1: "#db0e9a",
    2: "#938e7b",
    3: "#9999ff",
    4: "#a97101",
    5: "#1553ae",
    6: "#194a26",
    7: "#46e483",
    8: "#f3a60d",
    9: "#660082",
    10: "#55ff00",
    11: "#fff30d",
    12: "#e4df7c",
    13: "#000000",
}


lut_colors = {
    1: "#ff0000",
    2: "#D2B48C",
    3: "#707070",
    4: "#a97101",
    5: "#1553ae",
    6: "#194a26",
    7: "#46e483",
    8: "#f3a60d",
    9: "#660082",
    10: "#55ff00",
    11: "#fff30d",
    12: "#e4df7c",
    13: "#000000",
}


lut_classes = {
    1: "building",
    2: "pervious surface",
    3: "impervious surface",
    4: "bare soil",
    5: "water",
    6: "coniferous",
    7: "deciduous",
    8: "brushwood",
    9: "vineyard",
    10: "herbaceous vegetation",
    11: "agricultural land",
    12: "plowed land",
    13: "other",
}

## Functions


def get_data_paths(path, filter):
    for path in Path(path).rglob(filter):
        yield path.resolve().as_posix()


def remapping(lut: dict, recover="color") -> dict:
    rem = lut.copy()
    for idx in [13, 14, 15, 16, 17, 18, 19]:
        del rem[idx]
    if recover == "color":
        rem[13] = "#000000"
    elif recover == "class":
        rem[13] = "other"
    return rem


def convert_to_color(arr_2d: np.ndarray, palette: dict = lut_colors) -> np.ndarray:
    rgb_palette = {
        k: tuple(int(i * 255) for i in hex2color(v)) for k, v in palette.items()
    }
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    for c, i in rgb_palette.items():
        m = arr_2d == c
        arr_3d[m] = i
    return arr_3d


def display_nomenclature() -> None:
    GS = matplotlib.gridspec.GridSpec(1, 2)
    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor("black")

    plt.figtext(
        0.73,
        0.92,
        "REDUCED (BASELINE) NOMENCLATURE",
        ha="center",
        va="top",
        fontsize=14,
        color="w",
    )
    plt.figtext(
        0.3, 0.92, "FULL NOMENCLATURE", ha="center", va="top", fontsize=14, color="w"
    )

    full_nom = matplotlib.gridspec.GridSpecFromSubplotSpec(19, 1, subplot_spec=GS[0])
    for u, k in enumerate(lut_classes):
        curr_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=full_nom[u], width_ratios=[2, 6]
        )
        ax_color, ax_class = fig.add_subplot(
            curr_gs[0], xticks=[], yticks=[]
        ), fig.add_subplot(curr_gs[1], xticks=[], yticks=[])
        ax_color.set_facecolor(lut_colors[k])
        ax_class.text(
            0.05, 0.3, f"({u+1}) - " + lut_classes[k], fontsize=14, fontweight="bold"
        )
    main_nom = matplotlib.gridspec.GridSpecFromSubplotSpec(19, 1, subplot_spec=GS[1])
    for u, k in enumerate(remapping(lut_classes, recover="class")):
        curr_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=main_nom[u], width_ratios=[2, 6]
        )
        ax_color, ax_class = fig.add_subplot(
            curr_gs[0], xticks=[], yticks=[]
        ), fig.add_subplot(curr_gs[1], xticks=[], yticks=[])
        ax_color.set_facecolor(remapping(lut_colors, recover="color")[k])
        ax_class.text(
            0.05,
            0.3,
            f"({k}) - " + (remapping(lut_classes, recover="class")[k]),
            fontsize=14,
            fontweight="bold",
        )
    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_edgecolor("w"), spine.set_linewidth(1.5)
    plt.show()


def display_samples(images, masks, sentinel_imgs, centroid, palette=lut_colors) -> None:
    idx = random.sample(range(0, len(images)), 1)[0]
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
    fig.subplots_adjust(wspace=0.0, hspace=0.15)
    fig.patch.set_facecolor("black")

    with rasterio.open(images[idx], "r") as f:
        im = f.read([1, 2, 3]).swapaxes(0, 2).swapaxes(0, 1)
    with rasterio.open(masks[idx], "r") as f:
        print("masks[idx]:", masks[idx])
        print("idx:", idx)

        mk = f.read([1])
        mk = convert_to_color(mk[0], palette=palette)

    sen = np.load(sentinel_imgs[idx])[20, [2, 1, 0], :, :] / 2000
    sen_spatch = sen[
        :,
        centroid[idx][0] - int(20) : centroid[idx][0] + int(20),
        centroid[idx][1] - int(20) : centroid[idx][1] + int(20),
    ]
    transform = T.CenterCrop(10)
    sen_aerialpatch = transform(
        torch.as_tensor(np.expand_dims(sen_spatch, axis=0))
    ).numpy()
    sen = np.transpose(sen, (1, 2, 0))
    sen_spatch = np.transpose(sen_spatch, (1, 2, 0))
    sen_aerialpatch = np.transpose(sen_aerialpatch[0], (1, 2, 0))

    # axs = axs if isinstance(axs[], np.ndarray) else [axs]
    ax0 = axs[0][0]
    ax0.imshow(im)
    ax0.axis("off")
    ax1 = axs[0][1]
    ax1.imshow(mk, interpolation="nearest")
    ax1.axis("off")
    ax2 = axs[0][2]
    ax2.imshow(im)
    ax2.imshow(mk, interpolation="nearest", alpha=0.25)
    ax2.axis("off")
    ax3 = axs[1][0]
    ax3.imshow(sen)
    ax3.axis("off")
    ax4 = axs[1][1]
    ax4.imshow(sen_spatch)
    ax4.axis("off")
    ax5 = axs[1][2]
    ax5.imshow(sen_aerialpatch)
    ax5.axis("off")

    # Create a Rectangle patch
    rect = Rectangle(
        (centroid[idx][1] - 5.12, centroid[idx][0] - 5.12),
        10.24,
        10.24,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    ax3.add_patch(rect)
    rect = Rectangle(
        (14.88, 14.88), 10.24, 10.24, linewidth=1, edgecolor="r", facecolor="none"
    )
    ax4.add_patch(rect)

    ax0.set_title("RGB Image", size=12, fontweight="bold", c="w")
    ax1.set_title("Ground Truth Mask", size=12, fontweight="bold", c="w")
    ax2.set_title("Overlay Image & Mask", size=12, fontweight="bold", c="w")
    ax3.set_title("Sentinel super area", size=12, fontweight="bold", c="w")
    ax4.set_title("Sentinel super patch", size=12, fontweight="bold", c="w")
    ax5.set_title("Sentinel over the aerial patch", size=12, fontweight="bold", c="w")
    plt.show()


# def display_time_series(
#     sentinel_images, nb_samples, nb_dates=1
# ):
#     fig = plt.figure(figsize=(20, 20))
#     fig.patch.set_facecolor("black")
#     grid = ImageGrid(
#         fig,
#         111,  # similar to subplot(111)
#         nrows_ncols=(nb_samples, nb_dates),  # creates 2x2 grid of axes
#         axes_pad=0.25,  # pad between axes in inch.
#     )

#     for ax, im in zip(grid, sentinel_images):
#         # Iterating over the grid returns the Axes.
#         im = np.clip(im, 0, 1)
#         ax.imshow(im, aspect="auto")
#     plt.show()


def tensor2img(img):
    img = np.round((img + 1) * 127.5)
    img = np.clip(img, 0, 255)  # .astype(np.uint8)
    return img


def display_time_series(
    sentinel_images, clouds_masks, sentinel_products, nb_samples, nb_dates=5
):
    indices = random.sample(range(0, len(sentinel_images)), nb_samples)
    fig = plt.figure(figsize=(20, 20))
    fig.patch.set_facecolor("black")
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(nb_samples, nb_dates),  # creates 2x2 grid of axes
        axes_pad=0.25,  # pad between axes in inch.
    )

    img_to_plot = []

    for u, idx in enumerate(indices):
        img_dates = read_dates(sentinel_products[idx])

        sen = np.load(sentinel_images[idx])[:, [2, 1, 0], :, :] / 2000

        mask = np.load(clouds_masks[idx])

        print("sen", sen.shape)
        print("mask", mask.shape)

        # sen (46, 3, 309, 156)
        # mask (46, 2, 309, 156)

        dates_to_keep = filter_dates(sen, mask)
        if len(dates_to_keep) < 5:
            print("Not enough cloudless dates, not filtering")
            dates = random.sample(range(0, len(sen)), nb_dates)
        else:
            sen = sen[dates_to_keep]
            img_dates = img_dates[dates_to_keep]
            dates = random.sample(range(0, len(dates_to_keep)), nb_dates)

        for d in dates:
            sen_t = np.transpose(sen[d], (1, 2, 0))
            img_to_plot.append((sen_t, img_dates[d]))

    for ax, (im, date) in zip(grid, img_to_plot):
        # Iterating over the grid returns the Axes.
        im = np.clip(im, 0, 1)
        ax.imshow(im, aspect="auto")
        ax.set_title(date.strftime("%d/%m/%Y"), color="whitesmoke")
    plt.show()


def display_all(images, masks) -> None:
    GS = matplotlib.gridspec.GridSpec(20, 10, wspace=0.002, hspace=0.1)
    fig = plt.figure(figsize=(40, 100))
    fig.patch.set_facecolor("black")
    for u, k in enumerate(images):
        ax = fig.add_subplot(GS[u], xticks=[], yticks=[])
        with rasterio.open(k, "r") as f:
            img = f.read([1, 2, 3])
        rasterio.plot.show(img, ax=ax)
        ax.set_title(k.split("/")[-1][:-4], color="w")
        get_m = [i for i in masks if k.split("/")[-1].split("_")[1][:-4] in i][0]
        with rasterio.open(get_m, "r") as f:
            msk = f.read()
        ax.imshow(
            convert_to_color(msk[0], palette=lut_colors),
            interpolation="nearest",
            alpha=0.2,
        )
    plt.show()


def display_all_with_semantic_class(images, masks: list, semantic_class: int) -> None:
    def convert_to_color_and_mask(
        arr_2d: np.ndarray, semantic_class: int, palette: dict = lut_colors
    ) -> np.ndarray:
        rgb_palette = {
            k: tuple(int(i * 255) for i in hex2color(v)) for k, v in palette.items()
        }
        arr_3d = np.zeros((arr_2d[0].shape[0], arr_2d[0].shape[1], 4), dtype=np.uint8)
        for c, i in rgb_palette.items():
            m = arr_2d[0] == c
            if c == semantic_class:
                g = list(i)
                g.append(150)
                u = tuple(g)
                arr_3d[m] = u
            else:
                arr_3d[m] = tuple([0, 0, 0, 0])
        return arr_3d

    sel_imgs, sel_msks, sel_ids = [], [], []
    for img, msk in zip(images, masks):
        with rasterio.open(msk, "r") as f:
            data_msk = f.read()
        if semantic_class in list(set(data_msk.flatten())):
            sel_msks.append(
                convert_to_color_and_mask(data_msk, semantic_class, palette=lut_colors)
            )
            with rasterio.open(img, "r") as f:
                data_img = f.read([1, 2, 3])
            sel_imgs.append(data_img)
            sel_ids.append(img.split("/")[-1][:-4])
    if len(sel_imgs) == 0:
        print(
            "=" * 50,
            f"      SEMANTIC CLASS: {lut_classes[semantic_class]}",
            "...CONTAINS NO IMAGES IN THE CURRENT DATASET!...",
            "=" * 50,
            sep="\n",
        )
    else:
        print(
            "=" * 50,
            f"      SEMANTIC CLASS: {lut_classes[semantic_class]}",
            "=" * 50,
            sep="\n",
        )
        GS = matplotlib.gridspec.GridSpec(
            int(np.ceil(len(sel_imgs) / 5)), 5, wspace=0.002, hspace=0.15
        )
        fig = plt.figure(figsize=(30, 6 * int(np.ceil(len(sel_imgs) / 5))))
        fig.patch.set_facecolor("black")
        for u, (im, mk, na) in enumerate(zip(sel_imgs, sel_msks, sel_ids)):
            ax = fig.add_subplot(GS[u], xticks=[], yticks=[])
            ax.set_title(na, color="w")
            ax.imshow(im.swapaxes(0, 2).swapaxes(0, 1))
            ax.imshow(mk, interpolation="nearest")
        plt.show()


def display_predictions(
    images, predictions, nb_samples: int, palette=lut_colors, classes=lut_classes
) -> None:
    indices = random.sample(range(0, len(predictions)), nb_samples)
    fig, axs = plt.subplots(nrows=nb_samples, ncols=2, figsize=(17, nb_samples * 8))
    fig.subplots_adjust(wspace=0.0, hspace=0.01)
    fig.patch.set_facecolor("black")

    palette = remapping(palette, recover="color")
    classes = remapping(classes, recover="class")

    for u, idx in enumerate(indices):
        rgb_image = [i for i in images if predictions[idx].split("_")[-1][:-4] in i][0]
        with rasterio.open(rgb_image, "r") as f:
            im = f.read([1, 2, 3]).swapaxes(0, 2).swapaxes(0, 1)
        with rasterio.open(predictions[idx], "r") as f:
            mk = f.read([1]) + 1
            f_classes = np.array(list(set(mk.flatten())))
            mk = convert_to_color(mk[0], palette=palette)
        axs = axs if isinstance(axs[u], np.ndarray) else [axs]
        ax0 = axs[u][0]
        ax0.imshow(im)
        ax0.axis("off")
        ax1 = axs[u][1]
        ax1.imshow(mk, interpolation="nearest", alpha=1)
        ax1.axis("off")
        if u == 0:
            ax0.set_title("RGB Image", size=16, fontweight="bold", c="w")
            ax1.set_title("Prediction", size=16, fontweight="bold", c="w")
        handles = []
        for val in f_classes:
            handles.append(mpatches.Patch(color=palette[val], label=classes[val]))
        leg = ax1.legend(
            handles=handles,
            ncol=1,
            bbox_to_anchor=(1.4, 1.01),
            fontsize=12,
            facecolor="k",
        )
        for txt in leg.get_texts():
            txt.set_color("w")
        plt.show()


def filter_dates(
    img, mask, clouds: bool = 2, area_threshold: float = 0.5, proba_threshold: int = 20
):
    """Mask : array T*2*H*W
    Clouds : 1 if filter on cloud cover, 0 if filter on snow cover, 2 if filter on both
    Area_threshold : threshold on the surface covered by the clouds / snow
    Proba_threshold : threshold on the probability to consider the pixel covered (ex if proba of clouds of 30%, do we consider it in the covered surface or not)

    Return array of indexes to keep
    """
    dates_to_keep = []

    for t in range(mask.shape[0]):
        # Filter the images with only values above 1
        c = np.count_nonzero(img[t, :, :] > 1)
        if c != img[t, :, :].shape[1] * img[t, :, :].shape[2]:
            # filter the clouds / snow
            if clouds != 2:
                cover = np.count_nonzero(mask[t, clouds, :, :] >= proba_threshold)
            else:
                cover = np.count_nonzero(
                    (mask[t, 0, :, :] >= proba_threshold)
                ) + np.count_nonzero((mask[t, 1, :, :] >= proba_threshold))
            cover /= mask.shape[2] * mask.shape[3]
            if cover < area_threshold:
                dates_to_keep.append(t)
    return dates_to_keep


def read_dates(txt_file: str) -> np.array:
    with open(txt_file, "r") as f:
        products = f.read().splitlines()
    dates_arr = []
    for file in products:
        dates_arr.append(
            datetime.datetime(2021, int(file[15:19][:2]), int(file[15:19][2:]))
        )
    return np.array(dates_arr)


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


def display_downsampled_maps(images, masks, palette=lut_colors) -> None:
    idx = random.sample(range(0, len(images)), 1)[0]
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 6))
    fig.subplots_adjust(wspace=0.3, hspace=0.15)
    fig.patch.set_facecolor("black")

    with rasterio.open(images[idx], "r") as f:
        im = f.read([1, 2, 3]).swapaxes(0, 2).swapaxes(0, 1)
    with rasterio.open(masks[idx], "r") as f:
        mk = f.read([1])
        print("mk:", mk.shape)

    mk_down = torch.as_tensor(
        mk, dtype=torch.int32
    )  # .reshape(mk.shape[-1], mk.shape[0], mk.shape[1])

    print("mk tensor:", mk.shape)

    # mk_down = downsample_label_map(
    #     mk_down, output_size, padding_mode="replicate"
    # ).long()

    mk_down = downsample_majority_vote_with_crop(mk_down).long()

    print("mk_down downsampled:", mk_down.shape)

    mk = convert_to_color(mk[0], palette=palette)
    mk_down = convert_to_color(mk_down[0], palette=palette)

    print("mk convert_to_color:", mk.shape)
    print("mk_down convert_to_color:", mk_down.shape)

    # axs = axs if isinstance(axs[], np.ndarray) else [axs]
    ax0 = axs[0]
    ax0.imshow(im)
    ax0.axis("off")
    ax1 = axs[1]
    ax1.imshow(mk, interpolation="nearest")
    ax1.axis("off")
    ax2 = axs[2]
    ax2.imshow(im)
    ax2.imshow(mk_down, interpolation="nearest")
    ax2.axis("off")

    ax0.set_title("RGB Aerial Image", size=12, fontweight="bold", c="w")
    ax1.set_title("GT Mask", size=12, fontweight="bold", c="w")
    ax2.set_title("DownSampled GT Mask", size=12, fontweight="bold", c="w")

    # ax3.set_title("Sentinel super area", size=12, fontweight="bold", c="w")
    # ax4.set_title("Sentinel super patch", size=12, fontweight="bold", c="w")
    # ax5.set_title("Sentinel over the aerial patch", size=12, fontweight="bold", c="w")
    plt.show()


def downsample_vhr_aerial_image(
    input_tensor: torch.Tensor, downsample_factor: float
) -> torch.Tensor:
    """
    Downsamples a tensor representing an image from a higher resolution to a lower resolution.

    Args:
    - input_tensor (torch.Tensor): Tensor of shape (5, 512, 512) representing the image with 5 bands.
    - downsample_factor (float): Factor by which to downsample the image. E.g., 12.5 for downsampling from 20cm to 2.5m.

    Returns:
    - torch.Tensor: Downsampled tensor of shape (5, 256, 256).
    """

    print("img hr in :", input_tensor.shape)
    input_tensor = np.transpose(input_tensor, (2, 1, 0))
    print("img hr in permute :", input_tensor.shape)

    # Step 1: Calculate a valid kernel size
    kernel_size = int(2 * round(downsample_factor) + 1)  # Ensure kernel size is odd
    print(f"Kernel size used for Gaussian blur: {kernel_size}")

    # Step 2: Apply Gaussian Blur
    blur_transform = T.GaussianBlur(
        kernel_size=kernel_size, sigma=downsample_factor / 2
    )
    blurred_image = blur_transform(input_tensor).unsqueeze(0)  # Add batch dimension

    # Step 3: Downsample the Image
    # Compute the new size
    original_size = blurred_image.shape[2:]  # H, W of the image
    new_size = (
        int(original_size[0] / downsample_factor),
        int(original_size[1] / downsample_factor),
    )

    # Downsample the tensor
    downsampled_tensor_4d = F.interpolate(
        blurred_image, size=new_size, mode="bicubic", align_corners=False
    )

    # Remove the batch dimension, resulting in a shape of (5, 256, 256)
    downsampled_tensor = downsampled_tensor_4d.squeeze(0)
    downsampled_tensor = downsampled_tensor.permute(2, 1, 0)
    # downsampled_tensor = torch.clamp(downsampled_tensor, 0, 255)
    downsampled_tensor = torch.clamp(downsampled_tensor, min=0, max=255).to(torch.int32)

    # print("HR:", downsampled_tensor.shape) 40x40x3
    return downsampled_tensor


def upsample_sits_image(
    input_tensor: torch.Tensor, upsample_factor: int
) -> torch.Tensor:
    """
    Upsamples a tensor representing an image from a lower resolution to a higher resolution.

    Args:
    - input_tensor (torch.Tensor): Tensor of shape (5, 10, 10) representing the image with 5 bands.
    - upsample_factor (int): Factor by which to upsample the image. E.g., 4 for upsampling from 10m to 2.5m GSD.
    - final_size (tuple): The target size (height, width) for the output tensor. E.g., (256, 256).

    Returns:
    - torch.Tensor: Upsampled tensor of shape (5, 256, 256).
    """
    input_tensor = np.transpose(input_tensor, (2, 1, 0))
    # transform = T.CenterCrop((10, 10))
    # input_tensor = transform(input_tensor)
    # Calculate the intermediate size after upsampling by a factor of 'upsample_factor'
    new_height = input_tensor.shape[1] * upsample_factor
    new_width = input_tensor.shape[2] * upsample_factor

    # Step 1: Upsample the tensor to the intermediate size (after applying the upsample factor)
    input_tensor_4d = input_tensor.unsqueeze(0)  # Shape: (1, 5, 10, 10)

    # Upsample the tensor
    upsampled_tensor_4d = F.interpolate(
        input_tensor_4d,
        size=(new_height, new_width),
        mode="bicubic",
        align_corners=False,
    )

    # Remove the batch dimension (output shape will be (5, 256, 256))
    upsampled_tensor = upsampled_tensor_4d.squeeze(0)
    transform = T.CenterCrop((40, 40))

    # crop the LR to match HR
    cropped_upsampled_tensor = transform(upsampled_tensor)
    final_upsampled_tensor = cropped_upsampled_tensor.permute(2, 1, 0)
    # final_upsampled_tensor = np.clip(final_upsampled_tensor, 0, 1)
    final_upsampled_tensor = torch.clamp(final_upsampled_tensor, min=0, max=1)

    # print("UP:", final_upsampled_tensor.shape) 40x40x3
    return final_upsampled_tensor


def display_wow_samples(
    images, masks, sentinel_imgs, centroid, palette=lut_colors
) -> None:
    idx = random.sample(range(0, len(images)), 1)[0]
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
    fig.subplots_adjust(wspace=0.0, hspace=0.15)
    fig.patch.set_facecolor("black")

    # D:/kanyamahanga/Datasets/FLAIR_2_toy_dataset/flair_aerial_train/D016_2020/Z15_FN/img/IMG_013954.tif

    img_idx = "D:/kanyamahanga/Datasets/FLAIR/flair_aerial_test/D061_2020/Z3_AU/img/IMG_085221.tif"
    sen_idx = "D:/kanyamahanga/Datasets/FLAIR/flair_sen_test/D061_2020/Z3_AU/sen/SEN2_sp_D061_2020-Z3_AU_data.npy"
    msk_idx = "D:/kanyamahanga/Datasets/FLAIR/flair_labels_test/D061_2020/Z3_AU/msk/MSK_085221.tif"

    print(idx)
    print(images[idx])
    print(masks[idx])
    print(sentinel_imgs[idx])

    with rasterio.open(img_idx, "r") as f:
        im = f.read([1, 2, 3]).swapaxes(0, 2).swapaxes(0, 1)
    with rasterio.open(msk_idx, "r") as f:
        mk = f.read([1])
        mk = convert_to_color(mk[0], palette=palette)

    sen = np.load(sen_idx)
    print("---------------------------data:", sen.shape)
    sen = sen[20, [2, 1, 0], :, :] / 2000
    print("---------------------------data:", sen.shape)

    sen_spatch = sen[
        :,
        centroid[idx][0] - int(20) : centroid[idx][0] + int(20),
        centroid[idx][1] - int(20) : centroid[idx][1] + int(20),
    ]
    transform = T.CenterCrop(10)
    sen_aerialpatch = transform(
        torch.as_tensor(np.expand_dims(sen_spatch, axis=0))
    ).numpy()

    print(f"sen_aerialpatch: {sen_aerialpatch.shape}")

    sen = np.transpose(sen, (1, 2, 0))
    sen_spatch = np.transpose(sen_spatch, (1, 2, 0))

    # Get the minimum and maximum values
    min_val = im.min()
    max_val = im.max()

    print(f"Minimum value: {min_val}")
    print(f"Maximum value: {max_val}")

    sen_aerialpatch = np.transpose(sen_aerialpatch[0], (1, 2, 0))
    print("sen_aerialpatch:", sen_aerialpatch.shape)
    sen_spatch = torch.as_tensor(sen_spatch, dtype=torch.float)  # 40x40

    img_hr = torch.as_tensor(im, dtype=torch.float)
    # img_hr = im

    downsample_factor = 8  # Downsample factor from 20cm to 1.6m
    img_hr = downsample_vhr_aerial_image(img_hr, downsample_factor)

    # Upsamples a tensor representing an image from a lower resolution to a higher resolution.
    # print("lr_img_cropped:", sen_spatch.shape)
    img_lr_up = upsample_sits_image(sen_spatch, 4)
    print("LR:", sen_spatch.shape)

    # axs = axs if isinstance(axs[], np.ndarray) else [axs]
    ax0 = axs[0][0]
    ax0.imshow(im)
    ax0.axis("off")
    ax1 = axs[0][1]
    ax1.imshow(img_hr)
    ax1.axis("off")
    ax2 = axs[0][2]
    ax2.imshow(img_lr_up)
    ax2.axis("off")
    ax3 = axs[1][0]
    ax3.imshow(sen)
    ax3.axis("off")
    ax4 = axs[1][1]
    ax4.imshow(sen_spatch)
    ax4.axis("off")
    ax5 = axs[1][2]
    ax5.imshow(sen_aerialpatch)
    ax5.axis("off")

    # Create a Rectangle patch
    rect = Rectangle(
        (centroid[idx][1] - 5.12, centroid[idx][0] - 5.12),
        10.24,
        10.24,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    ax3.add_patch(rect)
    rect = Rectangle(
        (14.88, 14.88), 10.24, 10.24, linewidth=1, edgecolor="r", facecolor="none"
    )
    ax4.add_patch(rect)

    ax0.set_title("RGB Image", size=12, fontweight="bold", c="w")
    ax1.set_title("Downsampled RGB Image", size=12, fontweight="bold", c="w")
    ax2.set_title("Sentinel UP ", size=12, fontweight="bold", c="w")
    ax3.set_title("Sentinel LR", size=12, fontweight="bold", c="w")
    ax4.set_title("Sentinel super patch", size=12, fontweight="bold", c="w")
    ax5.set_title("Sentinel over the aerial patch", size=12, fontweight="bold", c="w")
    plt.show()


def display_wow_2_samples(
    images, masks, sentinel_imgs, centroid, palette=lut_colors
) -> None:
    idx = random.sample(range(0, len(images)), 1)[0]
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
    fig.subplots_adjust(wspace=0.0, hspace=0.15)
    fig.patch.set_facecolor("black")

    print(images[idx])

    with rasterio.open(images[idx], "r") as f:
        im = f.read([1, 2, 3]).swapaxes(0, 2).swapaxes(0, 1)  # hxwxb

    sen = np.load(sentinel_imgs[idx])[20, [2, 1, 0], :, :] / 2_000
    sen_spatch = sen[
        :,
        centroid[idx][0] - int(20) : centroid[idx][0] + int(20),
        centroid[idx][1] - int(20) : centroid[idx][1] + int(20),
    ]
    transform = T.CenterCrop(10)
    sen_aerialpatch = transform(
        torch.as_tensor(np.expand_dims(sen_spatch, axis=0))
    ).numpy()
    sen_aerialpatch = np.transpose(sen_aerialpatch[0], (1, 2, 0))

    sen = np.transpose(sen, (1, 2, 0))
    sen = np.clip(sen, 0, 1)

    sen_spatch = np.transpose(sen_spatch, (1, 2, 0))
    sen_spatch = np.clip(sen_spatch, 0, 1)

    print("sen:", sen.shape)  # (3x258x207)
    print(f"sen_spatch: {sen_spatch.shape}")  # (3x40x40)
    print(f"sen_aerialpatch: {sen_aerialpatch.shape}")  # (3x10x10)

    # Get the minimum and maximum values
    min_val_image_HR = im.min()
    max_val_image_HR = im.max()

    print(f"Minimum img value: {min_val_image_HR}")
    print(f"Maximum img value: {max_val_image_HR}")

    # Get the minimum and maximum values
    min_val_sen = sen.min()
    max_val_sen = sen.max()

    print(f"Minimum sen value: {min_val_sen}")
    print(f"Maximum sen value: {max_val_sen}")

    sen_spatch = torch.as_tensor(sen_spatch, dtype=torch.float)

    img_hr = torch.as_tensor(im, dtype=torch.float)
    # img_hr = im

    downsample_factor = 12.5  # Downsample factor from 20cm to 2.5m
    img_hr = downsample_vhr_aerial_image(img_hr, downsample_factor)

    # Upsamples a tensor representing an image from a lower resolution to a higher resolution.
    img_lr_up = upsample_sits_image(sen_spatch, 4)

    # Get the minimum and maximum values
    min_val_image_UP = img_lr_up.min()
    max_val_image_UP = img_lr_up.max()

    print(f"Minimum image_UP value: {min_val_image_UP}")
    print(f"Maximum image_UP value: {max_val_image_UP}")

    # Get the minimum and maximum values
    min_val_image_LR = sen_spatch.min()
    max_val_image_LR = sen_spatch.max()

    print(f"Minimum image_LR value: {min_val_image_LR}")
    print(f"Maximum image_LR value: {max_val_image_LR}")

    # Get the minimum and maximum values
    min_val_image_HR = img_hr.min()
    max_val_image_HR = img_hr.max()

    print(f"Minimum img_hr value: {min_val_image_HR}")
    print(f"Maximum img_hr value: {max_val_image_HR}")

    # axs = axs if isinstance(axs[], np.ndarray) else [axs]
    ax0 = axs[0][0]
    ax0.imshow(im)
    ax0.axis("off")
    ax1 = axs[0][1]
    ax1.imshow(img_hr)
    ax1.axis("off")
    ax2 = axs[0][2]
    ax2.imshow(img_lr_up)
    ax2.axis("off")
    ax3 = axs[1][0]
    ax3.imshow(sen)
    ax3.axis("off")
    ax4 = axs[1][1]
    ax4.imshow(sen_spatch)
    ax4.axis("off")
    ax5 = axs[1][2]
    ax5.imshow(sen_aerialpatch)
    ax5.axis("off")

    # Create a Rectangle patch
    rect = Rectangle(
        (centroid[idx][1] - 5.12, centroid[idx][0] - 5.12),
        10.24,
        10.24,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    ax3.add_patch(rect)
    rect = Rectangle(
        (14.88, 14.88), 10.24, 10.24, linewidth=1, edgecolor="r", facecolor="none"
    )
    ax4.add_patch(rect)

    gen_dir = "./"

    # im = tensor2img(im)
    img_lr_up = tensor2img(img_lr_up)
    sen = tensor2img(sen)
    sen_aerialpatch = tensor2img(sen_aerialpatch)

    # # Get the minimum and maximum values
    # min_val_image_HR = im.min()
    # max_val_image_HR = im.max()

    # print(f"Minimum im value: {min_val_image_HR}")
    # print(f"Maximum im value: {max_val_image_HR}")

    im = Image.fromarray(np.array(im).astype(np.uint8))
    img_lr_up = Image.fromarray(np.array(img_lr_up).astype(np.uint8))
    sen = Image.fromarray(np.array(sen).astype(np.uint8))
    sen_aerialpatch = Image.fromarray(np.array(sen_aerialpatch).astype(np.uint8))

    im.save(f"{gen_dir}/outputs/im.tiff", compression="tiff_lzw")
    img_lr_up.save(f"{gen_dir}/outputs/img_lr_up.tiff", compression="tiff_lzw")
    sen.save(f"{gen_dir}/outputs/sen.tiff", compression="tiff_lzw")
    sen_aerialpatch.save(
        f"{gen_dir}/outputs/sen_aerialpatch.tiff", compression="tiff_lzw"
    )

    ax0.set_title("RGB Image", size=12, fontweight="bold", c="w")
    ax1.set_title("Downsampled RGB Image", size=12, fontweight="bold", c="w")
    ax2.set_title("UP ", size=12, fontweight="bold", c="w")
    ax3.set_title("Sentinel super area", size=12, fontweight="bold", c="w")
    ax4.set_title("LR", size=12, fontweight="bold", c="w")
    ax5.set_title("LR Crop Aerial", size=12, fontweight="bold", c="w")
    plt.show()

    img = "./outputs/im.tiff"
    img_lr_up = "./outputs/img_lr_up.tiff"
    sen = "./outputs/sen.tiff"
    sen_aerialpatch = "./outputs/sen_aerialpatch.tiff"

    with rasterio.open(img, "r") as f:
        im = f.read([1, 2, 3]).swapaxes(0, 2).swapaxes(0, 1)  # hxwxb

        # Get the minimum and maximum values
        min_val_image_HR = im.min()
        max_val_image_HR = im.max()

        print(f"Minimum im: {min_val_image_HR}")
        print(f"Maximum im: {max_val_image_HR}")

    with rasterio.open(img_lr_up, "r") as f:
        img_lr_up = f.read([1, 2, 3]).swapaxes(0, 2).swapaxes(0, 1)  # hxwxb

        min_val_image_HR = img_lr_up.min()
        max_val_image_HR = img_lr_up.max()

        print(f"Minimum img_lr_up: {min_val_image_HR}")
        print(f"Maximum img_lr_up: {max_val_image_HR}")

    with rasterio.open(sen, "r") as f:
        sen = f.read([1, 2, 3]).swapaxes(0, 2).swapaxes(0, 1)  # hxwxb

    with rasterio.open(sen_aerialpatch, "r") as f:
        sen_aerialpatch = f.read([1, 2, 3]).swapaxes(0, 2).swapaxes(0, 1)  # hxwxb

    # Create a figure with a 2x2 grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
    fig.subplots_adjust(wspace=0.0, hspace=0.15)
    fig.patch.set_facecolor("black")

    ax0 = axes[0][0]
    ax0.imshow(im)
    ax0.axis("off")
    ax1 = axes[0][1]
    ax1.imshow(img_lr_up)
    ax1.axis("off")
    ax2 = axes[1][0]
    ax2.imshow(sen)
    ax2.axis("off")
    ax3 = axes[1][1]
    ax3.imshow(sen_aerialpatch)
    ax3.axis("off")
    # Adjust layout and display
    # plt.tight_layout()

    ax0.set_title("RGB Image", size=12, fontweight="bold", c="w")
    ax1.set_title("RGB Image UP", size=12, fontweight="bold", c="w")
    ax2.set_title("SEN", size=12, fontweight="bold", c="w")
    ax3.set_title("SEN AER PATCH", size=12, fontweight="bold", c="w")

    plt.show()


# D:/kanyamahanga/Datasets/FLAIR_2_toy_dataset/flair_aerial_train/D046_2019/Z24_UA/img/IMG_030679.tif
# D:/kanyamahanga/Datasets/FLAIR_2_toy_dataset/flair_aerial_train/D007_2020/Z33_UF/img/IMG_004683.tif

# D:/kanyamahanga/Datasets/FLAIR_2_toy_dataset/flair_aerial_test/D033_2021/Z20_UN/img/IMG_024977.tif
# D:/kanyamahanga/Datasets/FLAIR_2_toy_dataset/flair_aerial_test/D008_2019/Z17_UA/img/IMG_006975.tif
