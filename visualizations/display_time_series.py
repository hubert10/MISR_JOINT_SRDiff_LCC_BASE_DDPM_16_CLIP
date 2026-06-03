# Add Importing files from different parallel folder
import sys
import yaml

sys.path.append(
    "./"
)  # https://stackoverflow.com/questions/4383571/importing-files-from-different-folder

from pathlib import Path
from src.data_display import display_time_series
from src.load_data import load_data

config_path = "./flair-2-config.yml"  # Change to yours
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

display_time_series(sentinel_images, sentinel_masks, sentinel_products, nb_samples=3)
