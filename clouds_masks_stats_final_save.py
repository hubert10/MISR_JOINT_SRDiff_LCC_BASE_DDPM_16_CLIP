# Add Importing files from different parallel folder
import sys
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import torch
import torchvision.transforms as T
import numpy as np
from collections import defaultdict

sys.path.append(
    "./"
)  # https://stackoverflow.com/questions/4383571/importing-files-from-different-folder

from pathlib import Path
from data.data_display import display_time_series
from data.load_data import load_data

config_path = "./flair-config-server.yml"  # Change to yours
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Creation of the train, val and test dictionnaries with the data file paths
d_train, d_val, d_test = load_data(config)

images = d_train["PATH_IMG"]
labels = d_train["PATH_LABELS"]
sentinel_images = d_train["PATH_SP_DATA"]
clouds_masks = d_train["PATH_SP_MASKS"]  # Cloud masks
sentinel_products = d_train[
    "PATH_SP_DATES"
]  # Needed to get the dates of the sentinel images
centroids = d_train[
    "SP_COORDS"
]  # Position of the aerial image in the sentinel super area


output_dir = "output/cloud_stats"
os.makedirs(output_dir, exist_ok=True)


# ===== Provided Filtering Function =====
def filter_dates(
    img, mask, clouds: bool = 2, area_threshold: float = 0.5, proba_threshold: int = 60
):
    dates_to_keep = []

    for t in range(mask.shape[0]):
        c = np.count_nonzero(img[t] > 1)
        if c != img[t].shape[0] * img[t].shape[1]:
            if clouds != 2:
                cover = np.count_nonzero(mask[t, clouds, :, :] >= proba_threshold)
            else:
                cover = np.count_nonzero(
                    mask[t, 0, :, :] >= proba_threshold
                ) + np.count_nonzero(mask[t, 1, :, :] >= proba_threshold)
            cover /= mask.shape[2] * mask.shape[3]
            if cover < area_threshold:
                dates_to_keep.append(t)
    return dates_to_keep


# ===== Main Statistics Function with Filtering =====
def compute_cloud_coverage_statistics_with_filtering(
    sentinel_images, clouds_masks, centroids, crop_size=40, num_examples_to_display=5
):
    cc_thresholds = [0, 5, 10, 20, 30, 50, 70, 90, 100]

    cloud_coverage_per_image_before = []
    cloud_coverage_per_image_after = []
    cloud_coverage_per_series_flags = defaultdict(int)
    cloudy_examples = []

    for idx in range(len(sentinel_images)):
        try:
            sen = np.load(sentinel_images[idx])  # (T, C, H, W)
            mask = np.load(clouds_masks[idx])  # (T, 2, H, W)
        except Exception as e:
            print(f"Skipping series {idx} due to load error: {e}")
            continue

        if sen.ndim != 4 or mask.ndim != 4 or mask.shape[1] < 2:
            continue

        T_seq, C, H, W = sen.shape
        y, x = centroids[idx]
        y = np.clip(y, crop_size // 2, H - crop_size // 2)
        x = np.clip(x, crop_size // 2, W - crop_size // 2)

        ts_cloud_percents_before = []
        ts_cloud_percents_after = []

        # === Compute Before Filtering ===
        for t in range(T_seq):
            cloud_prob = mask[t, 1]  # Cloud probability (0-100)
            cropped_cloud = cloud_prob[
                y - crop_size // 2 : y + crop_size // 2,
                x - crop_size // 2 : x + crop_size // 2,
            ]
            cropped_cloud = T.CenterCrop(10)(
                torch.tensor(cropped_cloud.astype(np.uint8))
            ).numpy()
            binary_cloud_mask = (cropped_cloud >= 50).astype(np.uint8)
            cloud_percent = 100 * binary_cloud_mask.sum() / binary_cloud_mask.size

            cloud_coverage_per_image_before.append(cloud_percent)
            ts_cloud_percents_before.append(cloud_percent)

        # === Apply Filtering ===
        dates_to_keep = filter_dates(
            sen, mask, clouds=2, area_threshold=0.5, proba_threshold=60
        )

        if len(dates_to_keep) == 0:
            continue

        sen_filtered = sen[dates_to_keep]
        mask_filtered = mask[dates_to_keep]

        # === Compute After Filtering and Collect Examples ===
        for t_idx, t in enumerate(dates_to_keep):
            cloud_prob = mask_filtered[t_idx, 1]
            cropped_cloud = cloud_prob[
                y - crop_size // 2 : y + crop_size // 2,
                x - crop_size // 2 : x + crop_size // 2,
            ]
            cropped_cloud = T.CenterCrop(10)(
                torch.tensor(cropped_cloud.astype(np.uint8))
            ).numpy()

            binary_cloud_mask = (cropped_cloud >= 50).astype(np.uint8)
            cloud_percent = 100 * binary_cloud_mask.sum() / binary_cloud_mask.size

            cloud_coverage_per_image_after.append(cloud_percent)
            ts_cloud_percents_after.append(cloud_percent)

            cropped_img = sen_filtered[
                t_idx,
                [2, 1, 0],
                y - crop_size // 2 : y + crop_size // 2,
                x - crop_size // 2 : x + crop_size // 2,
            ]
            cropped_img = T.CenterCrop(10)(
                torch.tensor(cropped_img.astype(np.uint8))
            ).numpy()

            rgb_img = np.transpose(cropped_img / 2000.0, (1, 2, 0))
            rgb_img = np.clip(rgb_img, 0, 1)

            cloudy_examples.append((cloud_percent, rgb_img))

        for thresh in cc_thresholds:
            if any(cc >= thresh for cc in ts_cloud_percents_after):
                cloud_coverage_per_series_flags[thresh] += 1

    cloudy_examples = sorted(cloudy_examples, key=lambda x: -x[0])[
        :num_examples_to_display
    ]

    return (
        cloud_coverage_per_image_before,
        cloud_coverage_per_image_after,
        cloud_coverage_per_series_flags,
        cloudy_examples,
    )


# ===== Plotting Functions =====


def plot_cloud_coverage_statistics(before, after, series_flags, output_dir):
    plt.figure(figsize=(10, 6))
    plt.hist(
        before,
        bins=15,
        alpha=0.6,
        label="Before Filtering",
        color="gray",
        edgecolor="black",
    )
    plt.hist(
        after,
        bins=15,
        alpha=0.6,
        label="After Filtering",
        color="skyblue",
        edgecolor="black",
    )
    plt.xlabel("Cloud Coverage per Image (%)")
    plt.ylabel("Number of Images")
    plt.title("Image-Level Cloud Coverage (Before vs After Filtering)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "image_level_cloud_coverage.png"))
    plt.close()

    thresholds = sorted(series_flags.keys())
    counts = [series_flags[t] for t in thresholds]

    plt.figure(figsize=(8, 5))
    plt.bar(thresholds, counts, width=4, color="salmon", edgecolor="black")
    plt.xlabel("Cloud Coverage Threshold (%)")
    plt.ylabel("Number of Time Series")
    plt.title("Time Series-Level Cloud Coverage After Filtering")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "series_level_cloud_coverage.png"))
    plt.close()


def display_most_cloudy_examples(cloudy_examples, output_dir):
    num_examples = len(cloudy_examples)
    if num_examples == 0:
        print("⚠️ No cloudy examples to save.")
        return

    plt.figure(figsize=(4 * num_examples, 4))
    for idx, (cloud_percent, img) in enumerate(cloudy_examples):
        img_vis = img / np.percentile(img, 98)
        img_vis = np.clip(img_vis, 0, 1)

        plt.subplot(1, num_examples, idx + 1)
        plt.imshow(img_vis)
        plt.title(f"Cloud: {cloud_percent:.1f}%", fontsize=10)
        plt.axis("off")

        # Save individual image
        img_path = os.path.join(
            output_dir, f"cloudy_example_{idx+1}_{int(cloud_percent)}.png"
        )
        plt.imsave(img_path, img_vis)

    plt.suptitle("Top Most Cloudy Images After Filtering", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "most_cloudy_examples.png"))
    plt.close()


# ===== Example Usage =====
# sentinel_images = ['path/to/SEN2_series_1.npy', 'path/to/SEN2_series_2.npy', ...]
# clouds_masks = ['path/to/mask_series_1.npy', 'path/to/mask_series_2.npy', ...]
# centroids = [(120, 130), (115, 118), ...]

# Compute
(
    cloud_coverage_per_image_before,
    cloud_coverage_per_image_after,
    cloud_per_series_flags,
    cloudy_examples,
) = compute_cloud_coverage_statistics_with_filtering(
    sentinel_images, clouds_masks, centroids, crop_size=40, num_examples_to_display=5
)

# Save plots
plot_cloud_coverage_statistics(
    cloud_coverage_per_image_before,
    cloud_coverage_per_image_after,
    cloud_per_series_flags,
    output_dir,
)

# Save top cloudy images
display_most_cloudy_examples(cloudy_examples, output_dir)
