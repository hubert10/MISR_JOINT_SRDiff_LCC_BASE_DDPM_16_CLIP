import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import random
import torch
import rasterio.plot as plot
from torchvision import transforms as T


def generate_random_int():
    return random.randint(1, 2)


# Example usage
random_number = generate_random_int()
print("Random number:", random_number)
prefix = str(random_number)
prefix = "0"  # For testing purposes


def crop_sits_image(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    - cropped upsampled image to (5, 100, 100).
    Returns:
    - torch.Tensor: Upsampled tensor of shape (5, 256, 256).
    """
    cropping_ration = int(input_tensor.shape[-1] / 4)
    transform = T.CenterCrop((cropping_ration, cropping_ration))
    cropped_tensor = transform(input_tensor)
    return cropped_tensor


def crop_sits_image_np(input_array: np.ndarray) -> np.ndarray:
    """
    Center crop a NumPy array to 1/4 of its original spatial dimensions.

    Args:
        input_array (np.ndarray): Input array of shape (C, H, W)

    Returns:
        np.ndarray: Center-cropped array of shape (C, H/4, W/4)
    """
    input_array = np.transpose(input_array, (2, 0, 1))
    print("input_array.shape", input_array.shape)
    C, H, W = input_array.shape
    crop_size = H // 4  # assuming H == W

    top = (H - crop_size) // 2
    left = (W - crop_size) // 2

    cropped = input_array[:, top : top + crop_size, left : left + crop_size]
    cropped = np.transpose(cropped, (1, 2, 0))
    return cropped


def load_images_from_folder(folder, prefix):
    images = []
    filenames = sorted([f for f in os.listdir(folder) if f.startswith(prefix)])

    for filename in filenames:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    return images


def plot_image_series(hr_image, lowres_images, bicubic_images, sr_images):
    n = len(lowres_images)
    fig, axs = plt.subplots(4, n, figsize=(3 * n, 12))
    titles = ["HR", "LR", "Bic", "SR"]

    for col in range(n):
        # Row 1: HR image only in center
        for i in range(n):
            axs[0, i].axis("off")  # Clear all first-row axes
        center = n // 2
        axs[0, center].imshow(hr_image[0].squeeze())
        axs[0, center].axis("off")
        axs[0, center].set_title("HR")

        # Row 2: Low-resolution images
        # axs[1, col].imshow(lowres_images[col])
        axs[1, col].imshow(crop_sits_image_np(lowres_images[col]))
        axs[1, col].axis("off")
        axs[1, 0].set_ylabel(titles[1], fontsize=14)
        axs[1, col].set_title(f"Time {col+1}")

        # Row 3: Bicubic upsampled images
        axs[2, col].imshow(bicubic_images[col])
        axs[2, col].axis("off")
        axs[2, 0].set_ylabel(titles[2], fontsize=14)

        # Row 4: Super-resolved images
        axs[3, col].imshow(sr_images[col])
        axs[3, col].axis("off")
        axs[3, 0].set_ylabel(titles[3], fontsize=14)

    # plt.tight_layout()
    plt.show()


# ref_hr_image = (
#     "D:\\kanyamahanga\\Datasets\\MISR_S2_Aer_LCC_x10_MI_MO_Exp_Highresnet\\HR\\"
# )
# lowres_folder = (
#     "D:\\kanyamahanga\\Datasets\\MISR_S2_Aer_LCC_x10_MI_MO_Exp_Highresnet\\LR\\"
# )
# bicubic_folder = (
#     "D:\\kanyamahanga\\Datasets\\MISR_S2_Aer_LCC_x10_MI_MO_Exp_Highresnet\\UP\\"
# )
# sr_folder = "D:\\kanyamahanga\\Datasets\\MISR_S2_Aer_LCC_x10_MI_MO_Exp_Highresnet\\SR\\"


ref_hr_image = "C:\\Users\\kanyamahanga\Desktop\\IPI-128\\RESEARCH\\2025_Conference\\MISR_S2_Aer_LCC_X10_MI_MO_Exp_Loc\\checkpoints\\misr\\highresnet_ltae_ckpt\\results_0_\\HR\\"
lowres_folder = "C:\\Users\\kanyamahanga\Desktop\\IPI-128\\RESEARCH\\2025_Conference\\MISR_S2_Aer_LCC_X10_MI_MO_Exp_Loc\\checkpoints\\misr\\highresnet_ltae_ckpt\\results_0_\LR\\"
bicubic_folder = "C:\\Users\\kanyamahanga\Desktop\\IPI-128\\RESEARCH\\2025_Conference\\MISR_S2_Aer_LCC_X10_MI_MO_Exp_Loc\\checkpoints\\misr\\highresnet_ltae_ckpt\\results_0_\\UP\\"
sr_folder = "C:\\Users\\kanyamahanga\Desktop\\IPI-128\\RESEARCH\\2025_Conference\\MISR_S2_Aer_LCC_X10_MI_MO_Exp_Loc\\checkpoints\\misr\\highresnet_ltae_ckpt\\results_0_\\SR\\"


# === Load and Visualize ===
ref_hr_image = load_images_from_folder(ref_hr_image, prefix)
lowres_imgs = load_images_from_folder(lowres_folder, prefix)
bicubic_imgs = load_images_from_folder(bicubic_folder, prefix)
sr_imgs = load_images_from_folder(sr_folder, prefix)

assert (
    len(lowres_imgs) == len(bicubic_imgs) == len(sr_imgs)
), "Mismatch in time series length"

plot_image_series(ref_hr_image, lowres_imgs, bicubic_imgs, sr_imgs)


# id:  4936, 4479
