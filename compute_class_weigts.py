# Add Importing files from different parallel folder
import sys
import yaml

sys.path.append(
    "./"
)  # https://stackoverflow.com/questions/4383571/importing-files-from-different-folder

from pathlib import Path
from data.load_data import load_data
import numpy as np
import torch
import math
from sklearn import metrics
from pylab import *

from data.dataset import FitDataset
from data.load_data import load_data

config_path = "./flair-config-server.yml"  # Change to yours
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


def calc_cls_weights_log(dataset, config):
    """
    Function to calculate class weights depending on:
    https://medium.com/@meet_patel/notes-on-implementation-of-cross-entropy-loss-2a8e3408413c
    depending on all the labels that are available in the dataset
    :param dataset: pytorch dataset, based on class 'Eval_Dataset'
    :param config: All parameters that are used during training (from args.py)
    :return: cls_weights: np.array of size (number_of_classes) with the weights for each of the classes
                            example: [3.2, 1.5, 1.6, 3.9, 1.3]
    """
    anz_Pixel = 0
    cls_weights = np.zeros(13, dtype=np.float32)
    cls_percent = np.zeros(13, dtype=np.float32)

    for i in range(dataset.__len__()):
        current_ = dataset.__getitem__(i)
        current = torch.as_tensor(current_["labels"], dtype=torch.int32)

        # print("current:", type(current))

        current_np = current.numpy()
        # Totol number of pixels
        anz_Pixel = dataset.__len__() * current_np.shape[0] * current_np.shape[0]

        unique, counts = np.unique(current_np, return_counts=True)  # [0 1 2 3 4 5 6] []

        for j in range(len(unique)):
            clss = unique[j]
            cls_weights[clss] += counts[j]

    # print('occurences: ', cls_weights)
    # [ 105320.       0.  280732.       0.   40152.       0.  497384.  127176. 0.  200652. 1350892.       0.   19132.]
    # cls_percent = cls_weights / anz_Pixel
    # [0.04017639 0.         0.10709076 0.         0.01531677 0. 0.18973694 0.04851379 0.         0.07654266 0.5153244  0. 0.00729828]
    # print('percent:', cls_percent)

    for i in range(len(cls_weights)):
        if cls_weights[i] != 0:
            cls_weights[i] = math.log(anz_Pixel / cls_weights[i])
        else:
            cls_weights[i] = 0
    max_weight = np.max(cls_weights)
    cls_weights = cls_weights / max_weight
    cls_weights[0] = 0.1
    return cls_weights


d_train, d_val, d_test = load_data(config)

d_train = FitDataset(dict_files=d_train, config=config)
class_weights = torch.from_numpy(calc_cls_weights_log(d_train, 12))
