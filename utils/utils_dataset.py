import os
import yaml
import shutil
import json
import torch
import numpy as np
from torch.nn import functional as F
import torchvision.transforms as transforms
from skimage import exposure
import torchvision.transforms as T
from skimage import img_as_float


def read_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def filter_dates(
    mask,
    clouds: int = 2,
    area_threshold: float = 0.5,
    proba_threshold: int = 60,
    ref_patch_size: int = 64 * 64,
):
    T, _, H, W = mask.shape
    patch_size = H * W

    # 🔹 automatic threshold scaling
    scale = np.sqrt(ref_patch_size / patch_size)
    effective_area_threshold = min(0.95, area_threshold * scale)

    dates_to_keep = []
    coverages = []

    for t in range(T):
        if clouds != 2:
            covered = mask[t, clouds] >= proba_threshold
        else:
            covered = (mask[t, 0] >= proba_threshold) | (mask[t, 1] >= proba_threshold)

        cover = np.count_nonzero(covered) / patch_size
        coverages.append(cover)

        if cover < effective_area_threshold:
            dates_to_keep.append(t)

    # 🔹 fallback: least cloudy date
    if not dates_to_keep:
        dates_to_keep = [int(np.argmin(coverages))]

    return dates_to_keep


def pad_tensor(x, l, pad_value=0):
    padlen = l - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)


def pad_collate_train(dict, pad_value=0):
    _imgs = [i["img_hr"] for i in dict]
    imgs = [i["img"] for i in dict]
    txt = [i["txt"] for i in dict]
    _sen = [i["img_lr"] for i in dict]
    img_lr_up = [i["img_lr_up"] for i in dict]
    closest_idx = [i["closest_idx"] for i in dict]
    dates = [i["dates"] for i in dict]
    _msks_sr = [i["labels_sr"] for i in dict]
    _msks = [i["labels"] for i in dict]

    _dates = [i["dates_encoding"] for i in dict]
    _mtd = [i["mtd"] for i in dict]

    sizes = [e.shape[0] for e in _sen]
    m = max(sizes)
    padded_data, padded_data_up, padded_dates = [], [], []

    if not all(s == m for s in sizes):
        for data, data_up, date in zip(_sen, img_lr_up, _dates):
            padded_data.append(pad_tensor(data, m, pad_value=pad_value))
            padded_data_up.append(pad_tensor(data_up, m, pad_value=pad_value))
            padded_dates.append(pad_tensor(date, m, pad_value=pad_value))
    else:
        padded_data = _sen
        padded_data_up = img_lr_up
        padded_dates = _dates

    batch = {
        "img_hr": torch.stack(_imgs, dim=0),
        "img": torch.stack(imgs, dim=0),
        "img_lr": torch.stack(padded_data, dim=0),
        "img_lr_up": torch.stack(padded_data_up, dim=0),
        "labels": torch.stack(_msks, dim=0),
        "labels_sr": torch.stack(_msks_sr, dim=0),
        "dates": dates,
        "dates_encoding": torch.stack(padded_dates, dim=0),
        "closest_idx": torch.Tensor(closest_idx),
        "txt": txt,
        "mtd": torch.stack(_mtd, dim=0),
    }
    return batch


def pad_collate_predict(dict, pad_value=0):
    _imgs = [i["img_hr"] for i in dict]
    imgs = [i["img"] for i in dict]
    txt = [i["txt"] for i in dict]
    _sen = [i["img_lr"] for i in dict]
    img_lr_up = [i["img_lr_up"] for i in dict]
    item_name = [i["item_name"] for i in dict]
    closest_idx = [i["closest_idx"] for i in dict]
    dates = [i["dates"] for i in dict]
    _dates = [i["dates_encoding"] for i in dict]
    _msks = [i["labels"] for i in dict]
    _msks_sr = [i["labels_sr"] for i in dict]
    _mtd = [i["mtd"] for i in dict]
    sizes = [e.shape[0] for e in _sen]

    m = max(sizes)
    padded_data, padded_data_up, padded_dates = [], [], []

    if not all(s == m for s in sizes):
        for data, data_up, date in zip(_sen, img_lr_up, _dates):
            padded_data.append(pad_tensor(data, m, pad_value=pad_value))
            padded_data_up.append(pad_tensor(data_up, m, pad_value=pad_value))
            padded_dates.append(pad_tensor(date, m, pad_value=pad_value))
    else:
        padded_data = _sen
        padded_data_up = img_lr_up
        padded_dates = _dates

    batch = {
        "img_hr": torch.stack(_imgs, dim=0),
        "img": torch.stack(imgs, dim=0),
        "img_lr": torch.stack(padded_data, dim=0),
        "img_lr_up": torch.stack(padded_data_up, dim=0),
        "labels": torch.stack(_msks, dim=0),
        "labels_sr": torch.stack(_msks_sr, dim=0),
        "dates": dates,
        "dates_encoding": torch.stack(padded_dates, dim=0),
        "mtd": torch.stack(_mtd, dim=0),
        "txt": txt,
        "item_name": item_name,
        "closest_idx": torch.Tensor(closest_idx),
    }
    return batch


def date_to_day_of_year(given_date):
    """Convert a given date to day of year"""
    from datetime import datetime

    try:
        # string date format
        day_of_year = datetime.strptime(given_date, "%Y-%m-%d").timetuple().tm_yday
    except TypeError:
        # date type format
        day_of_year = given_date.timetuple().tm_yday

    # print("\nDay of year: ", day_of_year, "\n")
    return day_of_year


def sat_min_max_norm(img: torch.Tensor) -> torch.Tensor:
    """Image normalization from [min, max] to [-1,1]
    Min and max are computed across all channels
    """
    min_val = torch.min(img)
    max_val = torch.max(img)
    normalized_img = (img - min_val) / (max_val - min_val)
    transform = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            )
        ]
    )
    return transform(normalized_img).float()


def minmax(img: torch.Tensor) -> torch.Tensor:
    """Image normalization from [min, max] to [-1,1]
    Min and max are computed across all channels
    """
    min_val = torch.min(img)
    max_val = torch.max(img)
    normalized_img = (img - min_val) / (max_val - min_val)
    transform = transforms.Compose(
        [transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    )
    return transform(normalized_img).float()


def load_stats_file(path_file):
    """
    Load sentinel metadata from a file.

    Args:
        path_sen_metadata (str): Path to the sentinel metadata file.

    Returns:
        list: Loaded sentinel metadata.
    """
    if not os.path.exists(path_file):
        stats = []
    else:
        with open(path_file, "r") as f:
            stats = json.load(f)
    return stats


def sat_stretch_standardize(img: torch.Tensor, type: str, config: dict) -> torch.Tensor:
    assert type in ["aerial", "sen2"]

    path_aer_stats = os.path.join("./data/", "NORM_aer_patch.json")
    aer_stats = load_stats_file(path_aer_stats)
    if type == "aerial":
        # Normalization for Aerial Images
        transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=aer_stats["mean"][: config["num_channels_aer"]],
                    std=aer_stats["std"][: config["num_channels_aer"]],
                )
            ]
        )
        return 2 * transform(img).float() - 1
        # return transform(img).float()

    elif type == "sen2":
        # Normalization for Sentinel-2: Loop over all images in the series
        for e in range(img.shape[0]):
            # img[e] = stretch_standardize_utils(img[e], config)
            img[e] = sat_scaling_percentile(img[e])

        return img


def stretch_standardize_utils(img: torch.Tensor, config: dict) -> torch.Tensor:
    path_sat_stats = os.path.join("./data/", "NORM_sat_patch.json")
    sat_stats = load_stats_file(path_sat_stats)

    transform = transforms.Compose(
        [
            transforms.Normalize(
                mean=sat_stats["mean"][: config["num_channels_sat"]],
                std=sat_stats["std"][: config["num_channels_sat"]],
            )
        ]
    )
    # return 2 * transform(img).float() - 1
    return transform(img).float()


def sat_scaling_percentile(img: torch.Tensor) -> torch.Tensor:
    # min = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # max = np.array([24525.0, 22089.0, 19804.0, 18909.0, 19181.0, 18806.0, 18258.0, 18208.0, 15905.0, 15565.0])  # 98th percentile on train set

    min = np.array([0.0, 0.0, 0.0])
    max = np.array([24525.0, 22089.0, 19804.0])  # 98th percentile on train set

    # Clamp image values in the autorhized min, max range
    for i in range(len(max)):
        img[i] = img[i].clamp(min[i], max[i])

    transform = transforms.Compose([transforms.Normalize(mean=min, std=max - min)])
    return 2 * transform(img).float() - 1


def sat_scaling_mean_std(img: torch.Tensor) -> torch.Tensor:
    # min = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # max = np.array([24525.0, 22089.0, 19804.0, 18909.0, 19181.0, 18806.0, 18258.0, 18208.0, 15905.0, 15565.0])  # 98th percentile on train set

    min = np.array([0.0, 0.0, 0.0])
    max = np.array([24525.0, 22089.0, 19804.0])  # 98th percentile on train set

    # Clamp image values in the autorhized min, max range
    for i in range(3):
        img[i] = img[i].clamp(min[i], max[i])

    transform = transforms.Compose([transforms.Normalize(mean=min, std=max - min)])
    return 2 * transform(img).float() - 1


""" Spectral matching algorithms """


# Normal Standardization over whole dataset
def normalize(sen2, aerial, max_s2_images=1):
    transform_spot = transforms.Compose(
        [transforms.Normalize(mean=[479.0, 537.0, 344.0], std=[430.0, 290.0, 229.0])]
    )
    # dynamically define transform to reflect shape of tensor
    trans_mean, trans_std = [78.0, 91.0, 62.0] * max_s2_images, [
        36.0,
        28.0,
        30.0,
    ] * max_s2_images
    transform_sen = transforms.Compose(
        [transforms.Normalize(mean=trans_mean, std=trans_std)]
    )
    # perform transform
    sen2 = transform_sen(sen2)
    aerial = transform_spot(aerial)
    return sen2, aerial


# HISTOGRAM MATCHING
def histogram(sen2, aerial, max_s2_images=None):
    # https://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.match_histograms
    # have to transpose so that multichannel understands the dimensions
    sen2, aerial = sen2.numpy(), aerial.numpy()  # turn to np from tensor
    sen2 = np.transpose(sen2, (1, 2, 0))
    aerial = np.transpose(aerial, (1, 2, 0))
    aerial = exposure.match_histograms(image=aerial, reference=sen2, channel_axis=2)
    aerial, sen2 = np.transpose(aerial, (2, 0, 1)), np.transpose(sen2, (2, 0, 1))
    aerial, sen2 = torch.Tensor(aerial), torch.Tensor(sen2)
    return aerial


# MOMENT MATCHING
def moment(sen2, aerial, max_s2_images=None):
    sen2, aerial = sen2.numpy(), aerial.numpy()
    c = 0
    for channel_sen, channel_spot in zip(sen2, aerial):
        c += 1
        # calculate stats
        sen2_mean = np.mean(channel_sen)
        aerial_mean = np.mean(channel_spot)
        sen2_stdev = np.std(channel_sen)
        aerial_stdev = np.std(channel_spot)

        # calculate moment per channel
        channel_result = (
            ((channel_spot - aerial_mean) / aerial_stdev) * sen2_stdev
        ) + sen2_mean

        # stack channels to single array
        if c == 1:
            aerial = channel_result
        else:
            aerial = np.dstack((aerial, channel_result))
        # transpose back to Cx..

    aerial = torch.Tensor(aerial.transpose((2, 0, 1)))
    return aerial


def save_image_to_nested_folder(
    img, image_path, folder, sensor, timestamp=None, base_dir="."
):
    """
    Splits the given image path and saves the image into
    a nested folder structure.

    Args:
        image_path (str): Original image path string
        (e.g., 'D008_2019/Z13_UA/img/IMG_006512.tif')
        base_dir (str): Base directory where folders
        should be created (default is current directory)

    Returns:
        str: The final path where the image is saved
    """

    # Step-by-step parsing
    parts = image_path.split("/")
    first_folder = parts[0]
    second_folder = parts[1]
    # filename = parts[-1].replace('IMG', 'SEN2').replace('.tif', '.png')
    filename = str(first_folder) + "-" + str(second_folder) + ".png"

    # Create folder structure
    target_dir = os.path.join(
        base_dir, str(folder), first_folder, second_folder, str(sensor)
    )
    os.makedirs(target_dir, exist_ok=True)

    # Save image
    if timestamp is None:
        img.save(f"{target_dir}/{filename}")
    else:
        img.save(f"{target_dir}/{timestamp}-{filename}")
    save_path = os.path.join(target_dir, filename)
    return save_path


def save_hr_image_to_nested_folder(
    img, image_path, folder, sensor, timestamp=None, base_dir="."
):
    """
    Splits the given image path and saves the image into
    a nested folder structure.

    Args:
        image_path (str): Original image path string
        (e.g., 'D008_2019/Z13_UA/img/IMG_006512.tif')
        base_dir (str): Base directory where folders
        should be created (default is current directory)

    Returns:
        str: The final path where the image is saved
    """

    # Step-by-step parsing
    parts = image_path.split("/")
    first_folder = parts[0]
    second_folder = parts[1]
    last_folder = parts[-1]

    # filename = parts[-1].replace('IMG', 'SEN2').replace('.tif', '.png')
    # filename = str(first_folder) + "-" + str(second_folder) + str(last_folder).replace(".tif", ".png")
    filename = str(last_folder).replace(".tif", ".png")

    # Create folder structure
    target_dir = os.path.join(
        base_dir, str(folder), first_folder, second_folder, str(sensor)
    )
    os.makedirs(target_dir, exist_ok=True)
    # os.makedirs(os.path.dirname(target_dir), exist_ok=True)

    # Save image
    if timestamp is None:
        img.save(f"{target_dir}/{filename}")
    else:
        img.save(f"{target_dir}/{timestamp}-{filename}")
    save_path = os.path.join(target_dir, filename)
    return save_path


def save_image_to_nested_folder(
    img, image_path, folder, sensor, timestamp=None, base_dir="."
):
    """
    Splits the given image path and saves the image into
    a nested folder structure.

    Args:
        image_path (str): Original image path string
        (e.g., 'D008_2019/Z13_UA/img/IMG_006512.tif')
        base_dir (str): Base directory where folders
        should be created (default is current directory)

    Returns:
        str: The final path where the image is saved
    """

    # Step-by-step parsing
    parts = image_path.split("/")
    first_folder = parts[0]
    second_folder = parts[1]
    last_folder = parts[-1]

    # filename = parts[-1].replace('IMG', 'SEN2').replace('.tif', '.png')
    # filename = str(first_folder) + "-" + str(second_folder) + ".png"
    filename = ".png"

    # Create folder structure
    target_dir = os.path.join(
        base_dir,
        str(folder),
        first_folder,
        second_folder,
        str(sensor),
        str(last_folder.split("_")[1].split(".")[0]),
    )
    os.makedirs(target_dir, exist_ok=True)
    # os.makedirs(os.path.dirname(target_dir), exist_ok=True)

    # Save image
    if timestamp is None:
        img.save(f"{target_dir}/{filename}")
    else:
        img.save(f"{target_dir}/{timestamp}-{filename}")
    save_path = os.path.join(target_dir, filename)
    return save_path


def downsample_sr_image(
    input_tensor: torch.Tensor, downsample_factor: float = 10.0
) -> torch.Tensor:
    """
    Downsamples a tensor representing a batch of high-resolution images (1m GSD)
    to a lower resolution (e.g., 10m GSD) using Gaussian blur + interpolation.

    Args:
    - input_tensor (torch.Tensor): Tensor of shape (B, C, H, W), e.g., (B, C, 100, 100).
    - downsample_factor (float): Factor by which to downsample the image. Default is 10.0.

    Returns:
    - torch.Tensor: Downsampled tensor of shape (B, C, H//factor, W//factor).
    """
    B, C, H, W = input_tensor.shape

    # Step 1: Apply Gaussian blur channel-wise
    kernel_size = int(2 * round(downsample_factor) + 1)  # odd kernel size
    blur = T.GaussianBlur(kernel_size=kernel_size, sigma=downsample_factor / 2)

    # Apply blur on each image in the batch
    blurred = torch.stack([blur(img) for img in input_tensor])  # (B, C, H, W)

    # Step 2: Downsample using bicubic interpolation
    new_H, new_W = int(H / downsample_factor), int(W / downsample_factor)
    downsampled = F.interpolate(
        blurred, size=(new_H, new_W), mode="bicubic", align_corners=False
    )

    return downsampled


# CLIP-friendly phrasing
CLASS_TO_PHRASE = {
    "building": "buildings",
    "pervious surface": "paths and open ground",
    "impervious surface": "roads and paved areas",
    "bare soil": "bare soil",
    "water": "water bodies",
    "coniferous": "coniferous forest",
    "deciduous": "deciduous forest",
    "brushwood": "shrubs and brushwood",
    "vineyard": "vineyards",
    "herbaceous vegetation": "grass and low vegetation",
    "agricultural land": "farmland",
    "plowed land": "plowed fields",
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
