import os
import torch
import json
import rasterio
import datetime
import numpy as np
import pandas as pd
import torch.nn as nn
from utils import utils_dataset
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn.functional as F
from skimage import img_as_float
from utils.utils_dataset import (
    filter_dates,
    date_to_day_of_year,
    sat_stretch_standardize,
    sat_min_max_norm,
    sat_scaling_mean_std,
    sat_scaling_percentile,
    lut_classes,
    CLASS_TO_PHRASE,
)
import albumentations as A


class FitDataset(Dataset):
    def __init__(self, dict_files, config, use_augmentation=None, use_txt=True):
        # Set args as object attributes
        self.list_imgs = np.array(dict_files["PATH_IMG"])
        self.list_imgs_sp = np.array(dict_files["PATH_SP_DATA"])
        self.list_sp_coords = np.array(dict_files["SP_COORDS"])
        self.list_sp_products = np.array(dict_files["PATH_SP_DATES"])
        self.list_sp_masks = np.array(dict_files["PATH_SP_MASKS"])
        self.list_labels = np.array(dict_files["PATH_LABELS"])
        self.list_img_dates = np.array(dict_files["PATH_IMG_DATE"])
        self.use_augmentation = use_augmentation
        self.use_metadata = config["use_metadata"]
        if self.use_metadata == True:
            self.list_metadata = np.array(dict_files["MTD_AERIAL"])
        self.ref_year = config["ref_year"]
        self.ref_date = config["ref_date"]
        self.sat_patch_size = config["sat_patch_size"]
        self.num_classes = config["num_classes"]
        self.filter_mask = config["filter_clouds"]
        self.average_month = config["average_month"]
        self.config = config
        self.use_txt = use_txt

        # Channel Standardization
        self.sat_stretch_standardize = sat_stretch_standardize
        self.sat_min_max_norm = sat_min_max_norm
        self.sat_scaling_mean_std = sat_scaling_mean_std
        # To be added, but replaced by mean and std
        self.sat_scaling_percentile = sat_scaling_percentile

    def sat_min_max_norm(self, img_lr):
        # Compute channel-wise min and max across all timesteps, height, and width
        channel_min = np.min(img_lr, axis=(0, 2, 3), keepdims=True)  # Min per channel
        channel_max = np.max(img_lr, axis=(0, 2, 3), keepdims=True)  # Max per channel

        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-7
        normalized_img = (img_lr - channel_min) / (channel_max - channel_min + epsilon)
        return normalized_img

    def sat_scale_norm(self, img_lr):
        # scale to [0, 1] and normalize to [-1, 1]
        scaled_img_lr = img_lr / self.config["sat_reflectance_train"]
        scaled_img_lr = torch.clamp(scaled_img_lr, 0, 1)
        return 2 * scaled_img_lr - 1

    def aer_scale_norm(self, img_hr):
        # scale [0, 1] to [-1, 1]
        return 2 * img_hr - 1

    def read_img(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_img:
            array = src_img.read()
            return array

    def read_labels(self, raster_file: str, pix_tokeep: int = 500) -> np.ndarray:
        with rasterio.open(raster_file) as src_label:
            labels = src_label.read()[0]
            labels[labels > self.num_classes] = self.num_classes
            labels = labels - 1
            return labels, labels

    def read_superarea_and_crop(
        self, numpy_file: str, idx_centroid: list
    ) -> np.ndarray:
        data = np.load(numpy_file, mmap_mode="r")

        # if the loaded file is not msks, Select RGB channels and perform normalization
        if data.shape[1] != 2:
            data = data[:, [2, 1, 0, 6], :, :]

        subset_sp = data[
            :,
            :,
            idx_centroid[0]
            - int(self.sat_patch_size / 2) : idx_centroid[0]
            + int(self.sat_patch_size / 2),
            idx_centroid[1]
            - int(self.sat_patch_size / 2) : idx_centroid[1]
            + int(self.sat_patch_size / 2),
        ]
        return subset_sp

    def read_dates(self, txt_file: str) -> np.array:
        with open(txt_file, "r") as f:
            products = f.read().splitlines()
        diff_dates = []
        dates_arr = []
        for file in products:
            diff_dates.append(
                (
                    datetime.datetime(
                        int(self.ref_year),
                        int(self.ref_date.split("-")[0]),
                        int(self.ref_date.split("-")[1]),
                    )
                    - datetime.datetime(
                        int(self.ref_year), int(file[15:19][:2]), int(file[15:19][2:])
                    )
                ).days
            )
            dates_arr.append(
                datetime.datetime(
                    int(file[11:15][:4]), int(file[15:19][:2]), int(file[15:19][2:])
                )
            )
        return np.array(diff_dates), np.array(dates_arr)

    def monthly_image(self, sp_patch, sp_raw_dates):
        """
        Computes a monthly average using cloudless dates.
        If no cloudless dates are available for a specific
        month, fewer than 12 images may be used as input to
        the U-TAE branch
        """
        average_patch, average_dates, sp_raw_filtered_dates = [], [], []
        month_range = pd.period_range(
            start=sp_raw_dates[0].strftime("%Y-%m-%d"),
            end=sp_raw_dates[-1].strftime("%Y-%m-%d"),
            freq="{}{}".format(str(self.config["nbts"]), "M"),
        )

        for m in month_range:
            month_dates = list(
                filter(
                    lambda i: (sp_raw_dates[i].month == m.month)
                    and (sp_raw_dates[i].year == m.year),
                    range(len(sp_raw_dates)),
                )
            )
            # Take 15th each average patch!
            date = datetime.datetime(int(m.year), int(m.month), 15)

            if len(month_dates) != 0:
                if self.config["sen_temp_reduc"] == "median":
                    average_patch.append(np.median(sp_patch[month_dates], axis=0))
                else:
                    average_patch.append(np.mean(sp_patch[month_dates], axis=0))
                # Here we are taking the doy of the averaged patches per month
                average_dates.append(date_to_day_of_year(date) / 365.0001)
                sp_raw_filtered_dates.append(date)

        return (
            np.array(average_patch),
            np.array(average_dates),
            np.array(sp_raw_filtered_dates),
        )

    # Function to select the middle image from a time series tensor
    def select_middle_image_from_tensor(self, image_tensor):
        """
        Selects the middle image from a time series tensor of shape (T, C, H, W).

        Args:
            image_tensor (torch.Tensor): A 4D tensor representing the time series of images (T, C, H, W).

        Returns:
            torch.Tensor: The middle image tensor of shape (C, H, W).
        """
        if not isinstance(image_tensor, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor.")

        if image_tensor.ndim != 4:
            raise ValueError("Input tensor must have 4 dimensions (T, C, H, W).")

        # Get the temporal dimension (T)
        T = image_tensor.size(0)

        if T == 0:
            raise ValueError("The time series tensor has no images (T=0).")

        # Calculate the middle index
        middle_index = (T - 1) // 2
        # Select the middle image
        middle_image = image_tensor[middle_index]
        return middle_image

    def downsample_vhr_aerial_image(
        self, input_tensor: torch.Tensor, downsample_factor: float
    ) -> torch.Tensor:
        """
        Downsamples a tensor representing an image from a higher resolution to
          a lower resolution.

        Args:
        - input_tensor (torch.Tensor): Tensor of shape (5, 512, 512) representing
        the image with 5 bands.
        - downsample_factor (float): Factor by which to downsample the image.
        E.g., 3.2 for downsampling from 20cm to 1.6m.

        Returns:
        - torch.Tensor: Downsampled tensor of shape (5, 64, 64).
        """

        # Step 1: Calculate a valid kernel size
        kernel_size = int(2 * round(downsample_factor) + 1)  # Ensure kernel size is odd

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
        downsampled_tensor = F.interpolate(
            blurred_image, size=new_size, mode="bicubic", align_corners=False
        )

        # Remove the batch dimension, resulting in a shape of (5, 256, 256)
        downsampled_tensor = downsampled_tensor.squeeze(0)
        return downsampled_tensor

    def crop_then_upsample_sits_image(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Crops the center of the input satellite image and then upsamples it.

        Args:
        - input_tensor (torch.Tensor): Tensor of shape (C, H, W), e.g., (3, 10, 10).

        Returns:
        - torch.Tensor: Cropped and upsampled tensor of shape (C, up_H, up_W),
        e.g., (3, 100, 100)
        """
        # Ensure 3D shape (C, H, W)
        assert input_tensor.ndim == 3, "Input tensor must be (C, H, W)"

        # Step 1: Crop the center region (1/4th spatial extent)
        H, W = input_tensor.shape[1:]
        crop_h, crop_w = H // 4, W // 4
        center_crop = T.CenterCrop((crop_h, crop_w))
        cropped_tensor = center_crop(input_tensor)  # shape: (C, crop_h, crop_w)

        # Step 2: Add batch dimension for interpolation
        cropped_tensor = cropped_tensor.unsqueeze(0)  # shape: (1, C, h, w)

        # Step 3: Upsample using bicubic interpolation
        upsampled_tensor = F.interpolate(
            cropped_tensor,
            size=(64, 64),  # Use fixed size for consistency, e.g., (64, 64)
            mode="bicubic",
            align_corners=False,
        )

        return upsampled_tensor.squeeze(0)  # shape: (C, up_H, up_W)

    def downsample_single_label_map_majority_vote(self, label: torch.Tensor):
        """
        Downsamples a single multi-class label map from 20 cm GSD to 1.6 m GSD
        using majority vote on non-overlapping 8x8 blocks.

        Args:
            label (torch.Tensor): Input label map of shape [H, W] at 20 cm resolution.
                                H and W must be divisible by 8.

        Returns:
            torch.Tensor: Downsampled label map at 1.6 m resolution,
                        shape [H//8, W//8].
        """
        scale_factor = 8
        H, W = label.shape
        assert H == W, "Input label must be square."
        assert (
            H % scale_factor == 0
        ), "Input size must be divisible by scale factor (8)."

        output_size = H // scale_factor

        # Reshape to [output_size, scale_factor, output_size, scale_factor]
        label_reshaped = label.view(
            output_size, scale_factor, output_size, scale_factor
        )

        # Permute to [output_size, output_size, scale_factor, scale_factor]
        label_permuted = label_reshaped.permute(0, 2, 1, 3)

        # Flatten block pixels → [output_size, output_size, scale_factor * scale_factor]
        label_flat = label_permuted.reshape(
            output_size, output_size, scale_factor * scale_factor
        )

        # Majority vote along last dimension
        mode, _ = torch.mode(label_flat, dim=-1)
        return mode  # shape: [output_size, output_size]

    def get_srdiff_augmentation(self):
        """
        Applies identical ReplayCompose augmentation to a Sentinel-2 time series and a high-res aerial image.

        Args:
            lr_series (List[np.ndarray]): List of low-res Sentinel-2 time series images (H x W x C).
            hr_image (np.ndarray): High-res aerial image (H x W x C).
            use_augmentation (bool): Whether to apply augmentations.

        Returns:
            Tuple[List[np.ndarray], np.ndarray]: Augmented LR time series and HR image.
        """
        # Shared transform: for both LR time series and HR image
        shared_transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # A.ShiftScaleRotate(
                #     shift_limit=0.05,
                #     scale_limit=0.1,
                #     rotate_limit=15,
                #     border_mode=0,
                #     p=0.5,
                # ),
            ]
        )

        return shared_transform

    def tensor_to_numpy(self, img):
        if isinstance(img, torch.Tensor):
            return img.permute(1, 2, 0).cpu().numpy()
        return img

    def numpy_to_tensor(self, img_np):
        return torch.from_numpy(img_np).permute(2, 0, 1).float()

    def annotated_image_to_clip_text(
        self,
        tif_path: str,
        lut_classes: dict,
        min_ratio: float = 0.01,
        max_classes: int = 6,
        image_type: str = "aerial",
    ) -> str:
        """
        Convert a single-band class-ID GeoTIFF into a CLIP-friendly text description.

        Args:
            tif_path: path to .tif label map (values 1–13)
            lut_classes: mapping {class_id: class_name}
            min_ratio: minimum pixel ratio for a class to be included
            max_classes: maximum number of classes to mention
            image_type: 'satellite' or 'aerial'

        Returns:
            Natural-language description string
        """

        # Read single-band raster
        with rasterio.open(tif_path) as src:
            label_map = src.read(1)

        h, w = label_map.shape
        total_pixels = h * w

        present = []

        for class_id, class_name in lut_classes.items():
            if class_name == "other":
                continue

            ratio = np.sum(label_map == class_id) / total_pixels

            if ratio >= min_ratio:
                present.append((class_name, ratio))

        # Sort by dominance
        present.sort(key=lambda x: -x[1])
        present = present[:max_classes]

        if not present:
            return f"an {image_type} image of land"

        phrases = [CLASS_TO_PHRASE[name] for name, _ in present]

        if len(phrases) == 1:
            content = phrases[0]
        else:
            content = ", ".join(phrases[:-1]) + " and " + phrases[-1]

        return f"an {image_type} image of land with {content}"

    def __len__(self):
        return len(self.list_imgs)

    def __getitem__(self, index):
        image_file = self.list_imgs[index]
        img = self.read_img(raster_file=image_file)
        img = img_as_float(img)

        # metadata aerial images

        if self.use_metadata == True:
            mtd = self.list_metadata[index]
        else:
            mtd = []

        # labels (+ resized to satellite resolution if asked)
        labels_file = self.list_labels[index]
        labels, _ = self.read_labels(raster_file=labels_file)

        if self.use_txt:
            txt = self.annotated_image_to_clip_text(
                tif_path=labels_file, lut_classes=lut_classes, min_ratio=0.02
            )
        else:
            txt = ""

        sp_file = self.list_imgs_sp[index]
        sp_file_coords = self.list_sp_coords[index]
        sp_file_products = self.list_sp_products[index]
        sp_file_mask = self.list_sp_masks[index]
        img_date = self.list_img_dates[index]

        sp_patch = self.read_superarea_and_crop(sp_file, sp_file_coords)
        _, sp_raw_dates = self.read_dates(sp_file_products)
        sp_mask = self.read_superarea_and_crop(sp_file_mask, sp_file_coords)
        sp_mask = sp_mask.astype(int)

        if self.filter_mask:
            dates_to_keep = filter_dates(sp_mask)
            sp_patch = sp_patch[dates_to_keep]
            # sp_dates = sp_dates[dates_to_keep]
            sp_raw_dates = sp_raw_dates[dates_to_keep]

        if self.average_month:
            sp_patch, _, sp_raw_monthly_dates = self.monthly_image(
                sp_patch, sp_raw_dates
            )
        else:
            sp_raw_monthly_dates = sp_raw_dates

        if self.use_augmentation:
            shared_transform = self.get_srdiff_augmentation()
            # Convert to uint8 if needed for Albumentations
            img_lr_aug = []

            for t in range(sp_patch.shape[0]):
                lr_transformed = shared_transform(
                    image=sp_patch[t].swapaxes(0, 2).swapaxes(0, 1)
                )
                img_lr_aug.append(self.numpy_to_tensor(lr_transformed["image"]))

            img_lr = torch.stack(img_lr_aug, dim=0)  # shape: (T, C, H, W)
            sample = {
                "image": img.swapaxes(0, 2).swapaxes(0, 1),
                "mask": labels[None, :, :].swapaxes(0, 2).swapaxes(0, 1),
            }
            transformed_sample = shared_transform(**sample)
            img, labels = (
                transformed_sample["image"].swapaxes(0, 2).swapaxes(1, 2).copy(),
                transformed_sample["mask"].swapaxes(0, 2).swapaxes(1, 2).copy()[0],
            )

        # we use the positional encoding of the dates as proposed in the paper
        from datetime import datetime

        aerial_date = datetime.strptime(img_date, "%Y-%m-%d")
        sp_dates = [
            (date_sen2 - aerial_date).days for date_sen2 in sp_raw_monthly_dates
        ]
        sp_patch = img_as_float(sp_patch)

        img_lr = torch.as_tensor(sp_patch, dtype=torch.float)
        img = torch.as_tensor(img, dtype=torch.float)

        # Normalize the data by scaling: [0,1] to [-1, +1]
        if self.config["norm_type"] == "scale":
            img_lr = self.sat_scale_norm(img_lr)
            img = self.aer_scale_norm(img)

        # Normalize the data by min-max scaling [0,1] to [-1, +1]
        elif self.config["norm_type"] == "min-max":
            img_lr = self.sat_min_max_norm(img_lr)
            img = self.aer_scale_norm(img)

        # Normalize the data by standardization with mean and stds
        else:
            # Standardize the data to have zero mean and std one
            img = self.aer_scale_norm(img)
            # img = self.sat_stretch_standardize(img, "aerial", config=self.config)
            img_lr = self.sat_stretch_standardize(img_lr, "sen2", config=self.config)

        # Get the closest Sits image to aerial images acquisitions
        # img_lr_selected = self.select_middle_image_from_tensor(img_lr)
        img_lr_outs = []

        for i in range(img_lr.shape[0]):
            img_lr_outs.append(self.crop_then_upsample_sits_image(img_lr[i, :, :, :]))
        # torch.Size([2, 3, 40, 40]): T=2, C=3, H=40, W=40
        img_lr_up = torch.stack(img_lr_outs, 0)

        # Downsample from 20cm to 1m (scaling factor = 3.2)

        downsample_factor = 1.6 / 0.2
        img_hr = self.downsample_vhr_aerial_image(img, downsample_factor)
        ind = np.argmin(np.abs(sp_dates))

        # Match the ground truth radiometry to the reference low resolution input
        # (reference LR inpt is the closest image to the HR)
        labels = torch.as_tensor(labels, dtype=torch.int32)

        # cropping the ground truth images
        labels_sr = self.downsample_single_label_map_majority_vote(labels).long()

        return {
            "img_hr": img_hr,
            "img": img,
            "img_lr": img_lr,
            "img_lr_up": torch.as_tensor(img_lr_up, dtype=torch.float),
            "dates": torch.as_tensor(sp_dates, dtype=torch.float),
            "labels_sr": labels_sr,
            "labels": labels,
            "dates_encoding": torch.as_tensor(sp_dates, dtype=torch.float),
            "closest_idx": ind,
            "txt": txt,
            "mtd": torch.as_tensor(mtd, dtype=torch.float),
        }


class PredictDataset(Dataset):
    def __init__(self, dict_files, config, use_txt=True):
        self.list_imgs = np.array(dict_files["PATH_IMG"])
        self.list_imgs_sp = np.array(dict_files["PATH_SP_DATA"])
        self.list_sp_coords = np.array(dict_files["SP_COORDS"])
        self.list_sp_products = np.array(dict_files["PATH_SP_DATES"])
        self.list_sp_masks = np.array(dict_files["PATH_SP_MASKS"])
        self.list_img_dates = np.array(dict_files["PATH_IMG_DATE"])
        self.use_metadata = config["use_metadata"]
        self.list_labels = np.array(dict_files["PATH_LABELS"])
        if self.use_metadata == True:
            self.list_metadata = np.array(dict_files["MTD_AERIAL"])
        self.ref_year = config["ref_year"]
        self.ref_date = config["ref_date"]
        self.sat_patch_size = config["sat_patch_size"]
        self.num_classes = config["num_classes"]
        self.filter_mask = config["filter_clouds"]
        self.average_month = config["average_month"]
        self.config = config
        self.use_txt = use_txt

        # Channel Standardization
        self.sat_stretch_standardize = sat_stretch_standardize
        self.sat_min_max_norm = sat_min_max_norm
        self.sat_scaling_mean_std = sat_scaling_mean_std
        # To be added, but replaced by mean and std
        self.sat_scaling_percentile = sat_scaling_percentile

    def img_normalize(self, img):
        # Compute channel-wise min and max across all timesteps, height, and width
        channel_min = np.min(img, axis=(0, 2, 3), keepdims=True)  # Min per channel
        channel_max = np.max(img, axis=(0, 2, 3), keepdims=True)  # Max per channel

        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-7
        normalized_img = (img - channel_min) / (channel_max - channel_min + epsilon)
        return normalized_img

    def sat_scale_norm(self, img_lr):
        # scale to [0, 1] and normalize to [-1, 1]
        scaled_img_lr = img_lr / self.config["sat_reflectance_infer"]
        scaled_img_lr = torch.clamp(scaled_img_lr, 0, 1)
        return 2 * scaled_img_lr - 1

    def aer_scale_norm(self, img_hr):
        # scale [0, 1] to [-1, 1]
        return 2 * img_hr - 1

    def read_img(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_img:
            array = src_img.read()
            return array

    def read_labels(self, raster_file: str, pix_tokeep: int = 500) -> np.ndarray:
        with rasterio.open(raster_file) as src_label:
            labels = src_label.read()[0]
            labels[labels > self.num_classes] = self.num_classes
            labels = labels - 1
            return labels, labels

    def read_superarea_and_crop(
        self, numpy_file: str, idx_centroid: list
    ) -> np.ndarray:
        data = np.load(numpy_file, mmap_mode="r")

        # if the loaded file is not msks, Select RGB channels and perform normalization
        if data.shape[1] != 2:
            # 1. Blue (B2 490nm), 2. Green (B3 560nm), 3. Red (B4 665nm),
            # 4. Red-Edge (B5 705nm), 5. Red-Edge2 (B6 470nm), 6. Red-Edge3 (B7 783nm),
            #  7. NIR (B8 842nm), 8. NIR-Red-Edge (B8a 865nm), 9. SWIR (B11 1610nm),
            #  10. SWIR2 (B12 2190nm)

            data = data[:, [2, 1, 0, 6], :, :]
        subset_sp = data[
            :,
            :,
            idx_centroid[0]
            - int(self.sat_patch_size / 2) : idx_centroid[0]
            + int(self.sat_patch_size / 2),
            idx_centroid[1]
            - int(self.sat_patch_size / 2) : idx_centroid[1]
            + int(self.sat_patch_size / 2),
        ]
        return subset_sp

    def read_dates(self, txt_file: str) -> np.array:
        with open(txt_file, "r") as f:
            products = f.read().splitlines()
        diff_dates = []
        dates_arr = []
        for file in products:
            diff_dates.append(
                (
                    datetime.datetime(
                        int(self.ref_year),
                        int(self.ref_date.split("-")[0]),
                        int(self.ref_date.split("-")[1]),
                    )
                    - datetime.datetime(
                        int(self.ref_year), int(file[15:19][:2]), int(file[15:19][2:])
                    )
                ).days
            )
            dates_arr.append(
                datetime.datetime(
                    int(file[11:15][:4]), int(file[15:19][:2]), int(file[15:19][2:])
                )
            )
        return np.array(diff_dates), np.array(dates_arr)

    def monthly_image(self, sp_patch, sp_raw_dates):
        """
        Computes a monthly average using cloudless dates.
        If no cloudless dates are available for a specific
        month, fewer than 12 images may be used as input to
        the U-TAE branch
        """
        average_patch, average_dates, sp_raw_filtered_dates = [], [], []
        month_range = pd.period_range(
            start=sp_raw_dates[0].strftime("%Y-%m-%d"),
            end=sp_raw_dates[-1].strftime("%Y-%m-%d"),
            freq="{}{}".format(str(self.config["nbts"]), "M"),
        )

        for m in month_range:
            month_dates = list(
                filter(
                    lambda i: (sp_raw_dates[i].month == m.month)
                    and (sp_raw_dates[i].year == m.year),
                    range(len(sp_raw_dates)),
                )
            )
            # Take 15th each average patch!
            date = datetime.datetime(int(m.year), int(m.month), 15)

            if len(month_dates) != 0:
                if self.config["sen_temp_reduc"] == "median":
                    average_patch.append(np.median(sp_patch[month_dates], axis=0))
                else:
                    average_patch.append(np.mean(sp_patch[month_dates], axis=0))
                # Here we are taking the doy of the averaged patches per month
                average_dates.append(date_to_day_of_year(date) / 365.0001)
                sp_raw_filtered_dates.append(date)

        return (
            np.array(average_patch),
            np.array(average_dates),
            np.array(sp_raw_filtered_dates),
        )

    # Function to select the middle image from a time series tensor
    def select_middle_image_from_tensor(self, image_tensor):
        """
        Selects the middle image from a time series tensor of shape (T, C, H, W).

        Args:
            image_tensor (torch.Tensor): A 4D tensor representing the time series of images (T, C, H, W).

        Returns:
            torch.Tensor: The middle image tensor of shape (C, H, W).
        """
        if not isinstance(image_tensor, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor.")

        if image_tensor.ndim != 4:
            raise ValueError("Input tensor must have 4 dimensions (T, C, H, W).")

        # Get the temporal dimension (T)
        T = image_tensor.size(0)

        if T == 0:
            raise ValueError("The time series tensor has no images (T=0).")

        # Calculate the middle index
        middle_index = (T - 1) // 2

        # Select the middle image
        middle_image = image_tensor[middle_index]
        return middle_image

    def downsample_vhr_aerial_image(
        self, input_tensor: torch.Tensor, downsample_factor: float
    ) -> torch.Tensor:
        """
        Downsamples a tensor representing an image from a higher resolution to
          a lower resolution.

        Args:
        - input_tensor (torch.Tensor): Tensor of shape (5, 512, 512) representing
        the image with 5 bands.
        - downsample_factor (float): Factor by which to downsample the image.
        E.g., 3.2 for downsampling from 20cm to 1.6m.

        Returns:
        - torch.Tensor: Downsampled tensor of shape (5, 64, 64).
        """

        # Step 1: Calculate a valid kernel size
        kernel_size = int(2 * round(downsample_factor) + 1)  # Ensure kernel size is odd

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
        downsampled_tensor = F.interpolate(
            blurred_image, size=new_size, mode="bicubic", align_corners=False
        )

        # Remove the batch dimension, resulting in a shape of (5, 256, 256)
        downsampled_tensor = downsampled_tensor.squeeze(0)
        return downsampled_tensor

    def crop_then_upsample_sits_image(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Crops the center of the input satellite image and then upsamples it.

        Args:
        - input_tensor (torch.Tensor): Tensor of shape (C, H, W), e.g., (3, 10, 10).
        - upsample_factor (int): Upsampling factor, e.g., 10 → (10x spatial resolution).

        Returns:
        - torch.Tensor: Cropped and upsampled tensor of shape (C, up_H, up_W), e.g.,
          (3, 100, 100)
        """
        # Ensure 3D shape (C, H, W)
        assert input_tensor.ndim == 3, "Input tensor must be (C, H, W)"

        # Step 1: Crop the center region (1/4th spatial extent)
        H, W = input_tensor.shape[1:]
        crop_h, crop_w = H // 4, W // 4
        center_crop = T.CenterCrop((crop_h, crop_w))
        cropped_tensor = center_crop(input_tensor)  # shape: (C, crop_h, crop_w)

        # Step 2: Add batch dimension for interpolation
        cropped_tensor = cropped_tensor.unsqueeze(0)  # shape: (1, C, h, w)

        # Step 3: Upsample using bicubic interpolation
        upsampled_tensor = F.interpolate(
            cropped_tensor,
            size=(64, 64),
            mode="bicubic",
            align_corners=False,
        )
        return upsampled_tensor.squeeze(0)  # shape: (C, up_H, up_W)

    def downsample_single_label_map_majority_vote(self, label: torch.Tensor):
        """
        Downsamples a single multi-class label map from 20 cm GSD to 1.6 m GSD
        using majority vote on non-overlapping 8x8 blocks.

        Args:
            label (torch.Tensor): Input label map of shape [H, W] at 20 cm resolution.
                                H and W must be divisible by 8.

        Returns:
            torch.Tensor: Downsampled label map at 1.6 m resolution,
                        shape [H//8, W//8].
        """
        scale_factor = 8
        H, W = label.shape
        assert H == W, "Input label must be square."
        assert (
            H % scale_factor == 0
        ), "Input size must be divisible by scale factor (8)."

        output_size = H // scale_factor

        # Reshape to [output_size, scale_factor, output_size, scale_factor]
        label_reshaped = label.view(
            output_size, scale_factor, output_size, scale_factor
        )

        # Permute to [output_size, output_size, scale_factor, scale_factor]
        label_permuted = label_reshaped.permute(0, 2, 1, 3)

        # Flatten block pixels → [output_size, output_size, scale_factor * scale_factor]
        label_flat = label_permuted.reshape(
            output_size, output_size, scale_factor * scale_factor
        )

        # Majority vote along last dimension
        mode, _ = torch.mode(label_flat, dim=-1)
        return mode  # shape: [output_size, output_size]

    def annotated_image_to_clip_text(
        self,
        tif_path: str,
        lut_classes: dict,
        min_ratio: float = 0.01,
        max_classes: int = 6,
        image_type: str = "aerial",
    ) -> str:
        """
        Convert a single-band class-ID GeoTIFF into a CLIP-friendly text description.

        Args:
            tif_path: path to .tif label map (values 1–13)
            lut_classes: mapping {class_id: class_name}
            min_ratio: minimum pixel ratio for a class to be included
            max_classes: maximum number of classes to mention
            image_type: 'satellite' or 'aerial'

        Returns:
            Natural-language description string
        """

        # Read single-band raster
        with rasterio.open(tif_path) as src:
            label_map = src.read(1)

        h, w = label_map.shape
        total_pixels = h * w

        present = []

        for class_id, class_name in lut_classes.items():
            if class_name == "other":
                continue

            ratio = np.sum(label_map == class_id) / total_pixels

            if ratio >= min_ratio:
                present.append((class_name, ratio))

        # Sort by dominance
        present.sort(key=lambda x: -x[1])
        present = present[:max_classes]

        if not present:
            return f"an {image_type} image of land"

        phrases = [CLASS_TO_PHRASE[name] for name, _ in present]

        if len(phrases) == 1:
            content = phrases[0]
        else:
            content = ", ".join(phrases[:-1]) + " and " + phrases[-1]

        return f"an {image_type} image of land with {content}"

    def __len__(self):
        return len(self.list_imgs)

    def __getitem__(self, index):
        image_file = self.list_imgs[index]
        img = self.read_img(raster_file=image_file)
        img = img_as_float(img)

        if self.use_metadata == True:
            mtd = self.list_metadata[index]
        else:
            mtd = []

        if self.use_txt:
            txt = self.annotated_image_to_clip_text(
                tif_path=labels_file,
                lut_classes=lut_classes,
                min_ratio=0.02,
            )
        else:
            txt = ""

        # labels (+ resized to satellite resolution if asked)
        labels_file = self.list_labels[index]
        labels, _ = self.read_labels(raster_file=labels_file)

        # Sentinel patch, dates and cloud / snow mask
        sp_file = self.list_imgs_sp[index]
        sp_file_coords = self.list_sp_coords[index]
        sp_file_products = self.list_sp_products[index]
        sp_file_mask = self.list_sp_masks[index]
        img_date = self.list_img_dates[index]

        sp_patch = self.read_superarea_and_crop(sp_file, sp_file_coords)
        _, sp_raw_dates = self.read_dates(sp_file_products)
        sp_mask = self.read_superarea_and_crop(sp_file_mask, sp_file_coords)
        sp_mask = sp_mask.astype(int)

        if self.filter_mask:
            dates_to_keep = filter_dates(sp_mask)
            sp_patch = sp_patch[dates_to_keep]
            # sp_dates = sp_dates[dates_to_keep]
            sp_raw_dates = sp_raw_dates[dates_to_keep]

        if self.average_month:
            sp_patch, _, sp_raw_monthly_dates = self.monthly_image(
                sp_patch, sp_raw_dates
            )
        else:
            sp_raw_monthly_dates = sp_raw_dates

        # we use the positional encoding of the dates as proposed in the paper
        from datetime import datetime

        aerial_date = datetime.strptime(img_date, "%Y-%m-%d")
        sp_dates = [
            (date_sen2 - aerial_date).days for date_sen2 in sp_raw_monthly_dates
        ]

        # from datetime import datetime

        # Convert all items to datetime objects (if not already), then format them
        dates = [
            (
                date_sen2
                if isinstance(date_sen2, datetime)
                else datetime.strptime(date_sen2, "%Y-%m-%d")
            ).strftime("%Y-%m-%d")
            for date_sen2 in sp_raw_monthly_dates
        ]

        sp_patch = img_as_float(sp_patch)
        img_lr = torch.as_tensor(sp_patch, dtype=torch.float)
        img = torch.as_tensor(img, dtype=torch.float)

        # Normalize the data by scaling: [0,1] to [-1, +1]
        if self.config["norm_type"] == "scale":
            img_lr = self.sat_scale_norm(img_lr)
            img = self.aer_scale_norm(img)

        # Normalize the data by min-max scaling [0,1] to [-1, +1]
        elif self.config["norm_type"] == "min-max":
            img_lr = self.sat_min_max_norm(img_lr)
            img = self.aer_scale_norm(img)

        # Normalize the data by standardization with mean and stds
        else:
            # Standardize the data to have zero mean and std one
            img = self.aer_scale_norm(img)
            # img = self.sat_stretch_standardize(img, "aerial", config=self.config)
            img_lr = self.sat_stretch_standardize(img_lr, "sen2", config=self.config)

        # Upsamples a tensor representing an image from a lower resolution to a higher resolution.
        # Get the closest Sits image to aerial images acquisitions
        # img_lr_selected = self.select_middle_image_from_tensor(img_lr)

        img_lr_outs = []
        for i in range(img_lr.shape[0]):
            img_lr_outs.append(self.crop_then_upsample_sits_image(img_lr[i, :, :, :]))
        img_lr_up = torch.stack(img_lr_outs, 0)
        ind = np.argmin(np.abs(sp_dates))

        # Downsample from 0.2m to 1m (scaling factor = 5)
        downsample_factor = 1.6 / 0.2
        img_hr = self.downsample_vhr_aerial_image(img, downsample_factor)

        labels = torch.as_tensor(labels, dtype=torch.int32)
        # cropping the ground truth images
        labels_sr = self.downsample_single_label_map_majority_vote(labels).long()
        print("/".join(image_file.split("/")[-4:]))
        return {
            "img_hr": img_hr,
            "img": img,
            "img_lr": img_lr,
            "img_lr_up": img_lr_up,
            "dates": dates,
            "labels_sr": labels_sr,
            "labels": labels,
            "dates_encoding": torch.as_tensor(sp_dates, dtype=torch.float),
            "mtd": torch.as_tensor(mtd, dtype=torch.float),
            "closest_idx": ind,
            "txt": txt,
            "item_name": "/".join(image_file.split("/")[-4:]),
        }
