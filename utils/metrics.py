import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from data.load_data import DATA_DIR


def downsample_single_label_map_majority_vote_with_crop(
    label: torch.Tensor, original_size=512, cropped_size=500, output_size=100
):
    """
    Downsamples a single multi-class label map using majority vote after cropping.

    Args:
        label (torch.Tensor): Input label map of shape [512, 512].
        original_size (int): Original spatial size (assumed square). Default is 512.
        cropped_size (int): Desired spatial size after cropping (assumed square). Default is 500.
        output_size (int): Desired output spatial size (assumed square). Default is 100.

    Returns:
        torch.Tensor: Downsampled label map of shape [100, 100].
    """
    H, W = label.shape
    assert (
        H == original_size and W == original_size
    ), f"Input label map must be of shape [{original_size}, {original_size}]"
    assert (
        cropped_size % output_size == 0
    ), f"cropped_size must be divisible by output_size. Got cropped_size={cropped_size}, output_size={output_size}"

    # Step 1: Center crop to [500, 500]
    crop_margin = (original_size - cropped_size) // 2
    label_cropped = label[
        crop_margin : crop_margin + cropped_size,
        crop_margin : crop_margin + cropped_size,
    ]

    # Step 2: Reshape to [output_size, block_size, output_size, block_size]
    block_size = cropped_size // output_size  # e.g., 50
    label_reshaped = label_cropped.view(
        output_size, block_size, output_size, block_size
    )

    # Step 3: Permute to [output_size, output_size, block_size, block_size]
    label_permuted = label_reshaped.permute(0, 2, 1, 3)

    # Step 4: Flatten block pixels → [output_size, output_size, block_size * block_size]
    label_flat = label_permuted.reshape(
        output_size, output_size, block_size * block_size
    )

    # Step 5: Compute mode along last dimension (majority vote)
    mode, _ = torch.mode(label_flat, dim=-1)

    return mode  # shape: [output_size, output_size]


def generate_miou(config: str, path_truth: str, path_pred: str) -> list:
    #################################################################################################
    def get_data_paths(path, filter):
        for path in Path(path).rglob(filter):
            yield path.resolve().as_posix()

    def calc_miou(cm_array):
        """
        Specifically, we calculate the per-patch confusion matrix and per-class
        Intersection over Union (IoU) without excluding pixels belonging to
        the 'other' class, even though they represent a marginal part of the test-set.
        However, when computing the mean IoU (mIoU), we do remove the IoU of the 'other'
        class due to its association with majority or lower quality level pixels or very
        underrepresented land cover.
        """
        m = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            ious = np.diag(cm_array) / (
                cm_array.sum(0) + cm_array.sum(1) - np.diag(cm_array)
            )
        m = np.nansum(ious[:-1]) / (np.logical_not(np.isnan(ious[:-1]))).sum()
        return m.astype(float), ious[:-1]

    #################################################################################################

    truth_images = sorted(
        list(get_data_paths(os.path.join(DATA_DIR, Path(path_truth)), "MSK*.tif")),
        key=lambda x: int(x.split("_")[-1][:-4]),
    )
    preds_images = sorted(
        list(get_data_paths(Path(path_pred), "PRED*.tif")),
        key=lambda x: int(x.split("_")[-1][:-4]),
    )

    if len(truth_images) != len(preds_images):
        print("[ERROR !] mismatch number of predictions and test files.")
        return

    elif (
        truth_images[0][-10:-4] != preds_images[0][-10:-4]
        or truth_images[-1][-10:-4] != preds_images[-1][-10:-4]
    ):
        print("[ERROR !] unsorted images and masks found ! Please check filenames.")
        return

    else:
        patch_confusion_matrices = []

        for u in range(len(truth_images)):
            print(" truth_images[u]:", truth_images[u])
            target = (
                np.array(Image.open(truth_images[u])) - 1
            )  # -1 as model predictions start at 0 and turth at 1.
            target[target > 12] = (
                12  ### remapping masks to reduced baseline nomenclature.
            )
            preds = np.array(Image.open(preds_images[u]))

            target = torch.from_numpy(target)
            # target = downsample_single_label_map_majority_vote_with_crop(target)
            target = target.detach().cpu().numpy()

            patch_confusion_matrices.append(
                confusion_matrix(
                    target.flatten(), preds.flatten(), labels=list(range(13))
                )
            )
        # confusion_matrix_path = Path(config["outputs"]["out_folder"])
        sum_confmat = np.sum(patch_confusion_matrices, axis=0)
        mIou, ious = calc_miou(sum_confmat)
        # fig = plt.figure()
        # plt.matshow(sum_confmat)
        # plt.title("Confusion Matrix for Classification")
        # plt.colorbar()
        # plt.ylabel("True Label")
        # plt.xlabel("Pred. Label")
        # plt.savefig(os.path.join(confusion_matrix_path, "confusion_matrix" + ".jpg"))
        return mIou, ious


def generate_mf1s(path_truth: str, path_pred: str) -> list:
    #################################################################################################
    def get_data_paths(path, filter):
        for path in Path(path).rglob(filter):
            yield path.resolve().as_posix()

    def get_confusion_metrics(confusion_matrix):
        """Computes confusion metrics out of a confusion matrix (N classes)
        Parameters
        ----------
        confusion_matrix : numpy.ndarray
            Confusion matrix [N x N]
        Returns
        -------
        metrics : dict
            a dictionary holding all computed metrics
        Notes
        -----
        Metrics are: 'percentages', 'precisions', 'recalls', 'f1s', 'mf1', 'oa'
        """
        tp = np.diag(confusion_matrix)
        tp_fn = np.sum(confusion_matrix, axis=0)
        tp_fp = np.sum(confusion_matrix, axis=1)

        has_no_rp = tp_fn == 0
        has_no_pp = tp_fp == 0

        tp_fn[has_no_rp] = 1
        tp_fp[has_no_pp] = 1

        percentages = tp_fn / np.sum(confusion_matrix)
        precisions = tp / tp_fp
        recalls = tp / tp_fn

        p_zero = precisions == 0
        precisions[p_zero] = 1

        f1s = 2 * (precisions * recalls) / (precisions + recalls)
        ious = tp / (tp_fn + tp_fp - tp)

        precisions[has_no_pp] *= 0.0
        precisions[p_zero] *= 0.0
        recalls[has_no_rp] *= 0.0

        f1s[p_zero] *= 0.0
        f1s[percentages == 0.0] = np.nan
        ious[percentages == 0.0] = np.nan

        mf1 = np.nanmean(f1s[:-1])
        miou = np.nanmean(ious[:-1])

        oa = np.trace(confusion_matrix) / np.sum(confusion_matrix)

        metrics = {
            "percentages": percentages,
            "precisions": precisions,
            "recalls": recalls,
            "f1s": f1s,
            "mf1": mf1,
            "ious": ious,
            "miou": miou,
            "oa": oa,
        }
        return metrics

    truth_images = sorted(
        list(get_data_paths(os.path.join(DATA_DIR, Path(path_truth)), "MSK*.tif")),
        key=lambda x: int(x.split("_")[-1][:-4]),
    )
    preds_images = sorted(
        list(get_data_paths(Path(path_pred), "PRED*.tif")),
        key=lambda x: int(x.split("_")[-1][:-4]),
    )

    if len(truth_images) != len(preds_images):
        print("[WARNING !] mismatch number of predictions and test files.")
    if (
        truth_images[0][-10:-4] != preds_images[0][-10:-4]
        or truth_images[-1][-10:-4] != preds_images[-1][-10:-4]
    ):
        print("[WARNING !] unsorted images and masks found ! Please check filenames.")

    patch_confusion_matrices = []

    for u in range(len(truth_images)):
        target = (
            np.array(Image.open(truth_images[u])) - 1
        )  # -1 as model predictions start at 0 and turth at 1.
        target[target > 12] = 12  ### remapping masks to reduced baseline nomenclature.
        preds = np.array(Image.open(preds_images[u]))

        target = torch.from_numpy(target)
        # target = downsample_single_label_map_majority_vote_with_crop(target)
        target = target.detach().cpu().numpy()

        patch_confusion_matrices.append(
            confusion_matrix(target.flatten(), preds.flatten(), labels=list(range(13)))
        )

    sum_confmat = np.sum(patch_confusion_matrices, axis=0)
    metrics = get_confusion_metrics(sum_confmat)
    return metrics["mf1"], metrics["f1s"], metrics["oa"]
