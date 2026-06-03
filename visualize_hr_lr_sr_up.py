import cv2
import random
import torch
import os
import rasterio
import rasterio.plot as plot
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # export OMP_NUM_THREADS=4
os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


def generate_random_int():
    return random.randint(1, 10000)


# Example usage
random_number = generate_random_int()
print("Random number:", random_number)
img = random_number

# image_HR = "D:\\kanyamahanga\\Datasets\\MISR_S2_Aerial_LCC_SERVER\\results_0_\\outputs\\{}[HR].png".format(img)
# image_UP = "D:\\kanyamahanga\\Datasets\\MISR_S2_Aerial_LCC_SERVER\\results_0_\\UP\\{}.png".format(img)
# image_SR = "D:\\kanyamahanga\\Datasets\\MISR_S2_Aerial_LCC_SERVER\\results_0_\\outputs\\{}[SR].png".format(img)
# image_LR = "D:\\kanyamahanga\\Datasets\\MISR_S2_Aerial_LCC_SERVER\\results_0_\\LR\\{}_1.png".format(img)

# image_HR = "D:\\kanyamahanga\\Datasets\\MISR_S2_Aerial_LCC_X10_MI_SO\\HR\\{}.png".format(
#     img
# )
# image_UP = (
#     "D:\\kanyamahanga\\Datasets\\MISR_S2_Aerial_LCC_X10_MI_SO\\UP\\{}.png".format(
#         img
#     )
# )
# image_SR = "D:\\kanyamahanga\\Datasets\\MISR_S2_Aerial_LCC_X10_MI_SO\\SR\\{}.png".format(
#     img
# )
# image_LR = "D:\\kanyamahanga\\Datasets\\MISR_S2_Aerial_LCC_X10_MI_SO\\LR\\{}.png".format(
#     img
# )

# image_HR = "D:\\kanyamahanga\\Datasets\\MISR_S2_Aerial_LCC_X10_MI_SO\\outputs\\{}[HR].png".format(img)
# image_UP = "D:\\kanyamahanga\\Datasets\\MISR_S2_Aerial_LCC_X10_MI_SO\\UP\\{}.png".format(img)
# image_SR = "D:\\kanyamahanga\\Datasets\\MISR_S2_Aerial_LCC_X10_MI_SO\\outputs\\{}[SR].png".format(img)
# image_LR = "D:\\kanyamahanga\\Datasets\\MISR_S2_Aerial_LCC_X10_MI_SO\\LR\\{}_2.png".format(img)


image_HR = "C:\\Users\\kanyamahanga\Desktop\\IPI-128\\RESEARCH\\2025_Conference\\MISR_S2_Aer_LCC_X10_MI_MO_Exp_Loc\\checkpoints\\misr\\highresnet_ltae_ckpt\\results_0_\\HR\\{}.png".format(
    img
)
image_LR = "C:\\Users\\kanyamahanga\Desktop\\IPI-128\\RESEARCH\\2025_Conference\\MISR_S2_Aer_LCC_X10_MI_MO_Exp_Loc\\checkpoints\\misr\\highresnet_ltae_ckpt\\results_0_\LR\\{}.png".format(
    img
)
image_UP = "C:\\Users\\kanyamahanga\Desktop\\IPI-128\\RESEARCH\\2025_Conference\\MISR_S2_Aer_LCC_X10_MI_MO_Exp_Loc\\checkpoints\\misr\\highresnet_ltae_ckpt\\results_0_\\UP\\{}.png".format(
    img
)
image_SR = "C:\\Users\\kanyamahanga\Desktop\\IPI-128\\RESEARCH\\2025_Conference\\MISR_S2_Aer_LCC_X10_MI_MO_Exp_Loc\\checkpoints\\misr\\highresnet_ltae_ckpt\\results_0_\\SR\\{}.png".format(
    img
)


# image_HR = "C:\\Users\\kanyamahanga\Desktop\\IPI-128\\RESEARCH\\2025_Conference\\MISR_S2_Aerial_LCC_X10_MI_SO\\checkpoints\\misr\\srdiff_highresnet_ltae_ckpt\\results_0_\\outputs\\{}[HR].tiff".format(img)
# image_UP = "C:\\Users\\kanyamahanga\Desktop\\IPI-128\\RESEARCH\\2025_Conference\\MISR_S2_Aerial_LCC_X10_MI_SO\\checkpoints\\misr\\srdiff_highresnet_ltae_ckpt\\results_0_\\UP\\{}.tiff".format(img)
# image_SR = "C:\\Users\\kanyamahanga\Desktop\\IPI-128\\RESEARCH\\2025_Conference\\MISR_S2_Aerial_LCC_X10_MI_SO\\checkpoints\\misr\\srdiff_highresnet_ltae_ckpt\\results_0_\\outputs\\{}[SR].tiff".format(img)
# image_LR = "C:\\Users\\kanyamahanga\Desktop\\IPI-128\\RESEARCH\\2025_Conference\\MISR_S2_Aerial_LCC_X10_MI_SO\\checkpoints\\misr\\srdiff_highresnet_ltae_ckpt\\results_0_\\LR\\{}_1.tiff".format(img)

from PIL import Image
import torchvision.transforms as transforms


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


def load_raster_as_tensor(file_path):
    with rasterio.open(file_path, "r") as f:
        im = f.read([1, 2, 3]).swapaxes(0, 2).swapaxes(0, 1)  # hxwxb
    return im


def load_image_as_tensor(file_path):
    # Open the image
    image = Image.open(file_path)  # .convert("RGB")  # Convert to RGB if needed
    # Define the transformation
    transform = transforms.ToTensor()  # Converts to a tensor with shape [C, H, W]
    # Apply the transformation
    tensor_image = transform(image)
    return tensor_image


# Load the two images
image_HR = load_image_as_tensor(image_HR).swapaxes(0, 2).swapaxes(0, 1)
image_UP = load_image_as_tensor(image_UP).swapaxes(0, 2).swapaxes(0, 1)
image_SR = load_image_as_tensor(image_SR).swapaxes(0, 2).swapaxes(0, 1)
image_LR = crop_sits_image(load_image_as_tensor(image_LR)).swapaxes(0, 2).swapaxes(0, 1)

# image_HR = torch.as_tensor(image_HR, dtype=torch.float)
# image_UP = torch.as_tensor(image_UP, dtype=torch.float)
# image_SR = torch.as_tensor(image_SR, dtype=torch.float)
# image_LR = torch.as_tensor(image_LR, dtype=torch.float)

# Create a figure with a 2x2 grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
fig.subplots_adjust(wspace=0.0, hspace=0.15)
fig.patch.set_facecolor("black")

# Get the minimum and maximum values
min_val_image_HR = image_HR.min()
max_val_image_HR = image_HR.max()

print(f"Minimum image_HR value: {min_val_image_HR}")
print(f"Maximum image_HR value: {max_val_image_HR}")

# Get the minimum and maximum values
min_val_image_UP = image_UP.min()
max_val_image_UP = image_UP.max()

print(f"Minimum image_UP value: {min_val_image_UP}")
print(f"Maximum image_UP value: {max_val_image_UP}")

# Get the minimum and maximum values
min_val_image_SR = image_SR.min()
max_val_image_SR = image_SR.max()

print(f"Minimum image_SR value: {min_val_image_SR}")
print(f"Maximum image_SR value: {max_val_image_SR}")

# print("image_LR shape:", image_LR.shape)
# image_LR = crop_sits_image(image_LR)

# # Get the minimum and maximum values
# min_val_image_LR = image_LR.min()
# max_val_image_LR = image_LR.max()

# print(f"Minimum image_LR value: {min_val_image_LR}")
# print(f"Maximum image_LR value: {max_val_image_LR}")

ax0 = axes[0][0]
ax0.imshow(image_HR)  # , cmap="viridis", vmin=0, vmax=255)
ax0.axis("off")
ax1 = axes[0][1]
ax1.imshow(image_UP)  # , cmap="viridis", vmin=0, vmax=255)
ax1.axis("off")
ax2 = axes[1][0]
ax2.imshow(image_SR)  # , cmap="viridis", vmin=0, vmax=255 )
ax2.axis("off")
ax3 = axes[1][1]
ax3.imshow(image_LR)  # , cmap="viridis", vmin=0, vmax=255)
ax3.axis("off")
# Adjust layout and display
# plt.tight_layout()

ax0.set_title("HR (1m GSD)", size=12, fontweight="bold", c="w")
ax1.set_title("Bicubic UP (1m GSD)", size=12, fontweight="bold", c="w")
ax2.set_title("SR (1m GSD)", size=12, fontweight="bold", c="w")
ax3.set_title("LR (10m GSD)", size=12, fontweight="bold", c="w")

plt.show()

# Examples of how to save the images
# id: 7900, 7285, 154, 9886, 8566, 9933, 8485, 1665, 4963


# Hallucinations
# id: 9366, 7281
