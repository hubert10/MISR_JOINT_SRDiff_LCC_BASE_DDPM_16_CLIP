## Imports
import os
import sys
import random
from pathlib import Path
import numpy as np
import yaml
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import rasterio
import rasterio.plot as plot
import torch
import torchvision.transforms as T
import datetime
import shutil

sys.path.append(
    "./"
)  # https://stackoverflow.com/questions/4383571/importing-files-from-different-folder

DATA_DIR = "D:\kanyamahanga\Datasets"
BIGWORK_DIR = "D:\kanyamahanga\Bigwork"
flair_aerial_path = DATA_DIR + "/FLAIR/flair_aerial_test/"
msk_folder_aerial = BIGWORK_DIR + "/MISR_JOINT_SRDiff_LCC_BASE_DDPM_16_CLIP/PR/"
pred_folder_aerial = BIGWORK_DIR + "/MISR_JOINT_SRDiff_LCC_BASE_DDPM_16_CLIP/results"

# path to destination folders
test_folder_aerial = os.path.join(flair_aerial_path, "D022_2021/Z14_UU/img")
target_path_aerial = os.path.join(pred_folder_aerial, "predictions_Z14_UU")

# Create a list of image filenames in 'flair_aerial_path'
imgs_list = [filename for filename in os.listdir(test_folder_aerial)]
imgs_list = [filename.split("_")[-1] for filename in imgs_list]

print(imgs_list)

msks_list = [
    filename
    for filename in os.listdir(msk_folder_aerial)
    if filename.split("_")[-1] in imgs_list
]

# Create destination folders if they don't exist
# for folder_path in [train_folder_aerial]:
if not os.path.exists(target_path_aerial):
    os.makedirs(target_path_aerial, mode=777)

# Copy image files to destination folders
for i, f in enumerate(msks_list):
    shutil.copyfile(
        os.path.join(msk_folder_aerial, f), os.path.join(target_path_aerial, f)
    )
