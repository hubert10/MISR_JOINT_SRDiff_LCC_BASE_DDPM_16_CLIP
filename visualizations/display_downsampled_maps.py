# Add Importing files from different parallel folder
import sys
import os
import yaml

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # export OMP_NUM_THREADS=4
os.environ["OMP_NUM_THREADS"] = "6"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "6"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "6"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"  # export NUMEXPR_NUM_THREADS=6

sys.path.append(
    "./"
)  # https://stackoverflow.com/questions/4383571/importing-files-from-different-folder

from pathlib import Path
from src.data.data_display import display_downsampled_maps
from src.data.load_data import load_data

config_path = "./flair-config.yml"  # Change to yours
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Creation of the train, val and test dictionnaries with the data file paths
d_train, d_val, d_test = load_data(config)

images = d_train["PATH_IMG"]
labels = d_train["PATH_LABELS"]
sentinel_images = d_train["PATH_SP_DATA"]
sentinel_masks = d_train["PATH_SP_MASKS"]  # Cloud masks
sentinel_products = d_train[
    "PATH_SP_DATES"
]  # Needed to get the dates of the sentinel images
centroids = d_train[
    "SP_COORDS"
]  # Position of the aerial image in the sentinel super area

display_downsampled_maps(images, labels)
