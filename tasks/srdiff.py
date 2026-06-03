import os.path
import json
import torch
import numpy as np
import torch.nn as nn
from trainer import Trainer
from utils.hparams import hparams
from utils.utils import load_ckpt

# from models.diffsr_modules import Unet
from models.denoiser.sat_unet import SatUNet
from models.diffusion.ddpm import GaussianDiffusion
from models.sits_aerial_seg_model import SITSAerialSegmenter
from losses.focal_smooth import FocalLossWithSmoothing
from models.encoders.t_convformer import TConvFormer


class SRDiffTrainer(Trainer):
    def build_model(self):
        hidden_size = hparams["hidden_size"]
        dim_mults = hparams["unet_dim_mults"]
        dim_mults = [int(x) for x in dim_mults.split("|")]

        self.criterion_aer = FocalLossWithSmoothing(
            hparams["num_classes"], gamma=2, alpha=1, lb_smooth=0.2
        )
        self.criterion_sat = FocalLossWithSmoothing(
            hparams["num_classes"], gamma=2, alpha=1, lb_smooth=0.2
        )

        cond_stage_config = {
            "image_size": 64,
            "in_channels": 8,
            "model_channels": 160,
            "out_channels": 4,
            "num_res_blocks": 2,
            "attention_resolutions": [16, 8],
            "channel_mult": [1, 2, 2, 4],
            "num_head_channels": 32,
        }
        self.loss_aux_sat_weight = hparams["loss_aux_sat_weight"]
        self.loss_main_sat_weight = hparams["loss_main_sat_weight"]

        # self.denoise_net = Unet(
        #     hidden_size,
        #     out_dim=hparams["num_channels_sat"],
        #     cond_dim=hparams["rrdb_num_feat"],
        #     dim_mults=dim_mults,
        # )

        # Load pretrained weights for model initialization
        self.denoise_net, loading_info = SatUNet.from_pretrained(
            hparams["diff_sat_weights"],
            subfolder="checkpoint-150000/unet",
            revision=None,
            num_metadata=7,
            use_metadata=True,
            low_cpu_mem_usage=False,
            output_loading_info=True,
        )
        print("=" * 80)
        print("[INFO] SatUNet pretrained checkpoint loaded successfully.")
        print("Missing:", loading_info["missing_keys"])
        print("Unexpected:", loading_info["unexpected_keys"])
        print("=" * 80)

        # 1. SITS Encoder
        self.cond_net = TConvFormer(
            input_size=(hparams["sat_patch_size"], hparams["sat_patch_size"]),
            stem_channels=64,
            block_channels=hparams["block_channels"][:2],  # [64, 128, 256, 512]
            block_layers=hparams["block_layers"][1:],  # [2, 2, 5],  # [2, 2, 5, 2]
            head_dim=32,
            stochastic_depth_prob=0.2,
            partition_size=4,
        )
        # The pretrained weights are loaded for the first time and as far as the
        # cond the cond_net params are not frozen, they should be finetuned as well

        if (
            hparams["cond_net_ckpt"] != ""
            and os.path.exists(hparams["cond_net_ckpt"])
            and not hparams["infer"]
        ):
            weights_path = hparams["cond_net_ckpt"]
            if torch.cuda.is_available():
                old_dict = torch.load(weights_path, weights_only=False)
            else:
                old_dict = torch.load(weights_path, map_location=torch.device("cpu"))
            model_dict = self.cond_net.state_dict()
            old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
            model_dict.update(old_dict)
            print(f"| load 'cond model' from '{weights_path}'.")
            self.cond_net.load_state_dict(model_dict)

        self.gaussian = GaussianDiffusion(
            denoise_net=self.denoise_net,
            cond_net=self.cond_net,
            timesteps=hparams["timesteps"],
            loss_type=hparams["loss_type"],
        )

        self.model = SITSAerialSegmenter(gaussian=self.gaussian, config=hparams)
        if hparams["infer"]:
            if hparams["diff_net_ckpt"] != "" and os.path.exists(
                hparams["diff_net_ckpt"]
            ):
                load_ckpt(self.model, hparams["diff_net_ckpt"])

        # what is used for?
        self.global_step = 0
        return self.model

    def training_step(self, batch):
        img = batch["img"]  # torch.Size([4, 5, 512, 512])
        img_hr = batch["img_hr"]  # torch.Size([4, 5, 512, 512])
        img_lr = batch["img_lr"]  # torch.Size([4, 2, 3, 40, 40])
        img_lr_up = batch["img_lr_up"]  # torch.Size([4, 2, 3, 160, 160])
        labels = batch["labels"]  # torch.Size([4, 2, 3, 160, 160])
        labels_sr = batch["labels_sr"]  # torch.Size([4, 2, 3, 160, 160])
        txt = batch["txt"]  # torch.Size([4, 12, 3, 64, 64])
        mtd = batch["mtd"]  # torch.Size([4, 19, 64, 64])
        dates = batch["dates_encoding"]
        closest_idx = batch["closest_idx"]  # torch.Size([4, 2, 3, 160, 160])
        sc_img_hr = img_hr[:, :4, :, :]

        if hparams["use_highresnet_ltae"]:
            # call gaussian diffusion model for SR-prediction this should also
            # return the SR-SITS images alongside the diffusion losses
            losses, _, _, img_sr = self.model.gaussian(
                img,
                txt,
                mtd,
                sc_img_hr,
                img_lr,
                img_lr_up,
                labels_sr,
                dates=dates,
                closest_idx=closest_idx,
                # config=self.config,
            )

            # for classification branches
            cls_sits, multi_outputs, aer_outputs = self.model(
                img, img_sr, labels, dates, hparams
            )

            # Auxiliary losses
            # The CE loss for the SITS classification branch is done at 1m GSD

            aux_loss1 = self.criterion_sat(multi_outputs[2], labels_sr)
            aux_loss2 = self.criterion_sat(multi_outputs[1], labels_sr)
            aux_loss3 = self.criterion_sat(multi_outputs[0], labels_sr)

            # loss for main SITS classification branch
            loss_main_sat = self.criterion_sat(cls_sits, labels_sr)

            # Total loss for SITS branch
            loss_sat = self.loss_main_sat_weight * loss_main_sat + (
                self.loss_aux_sat_weight * aux_loss1
                + self.loss_aux_sat_weight * aux_loss2
                + self.loss_aux_sat_weight * aux_loss3
            )

            # print("labels:", labels.shape)
            # print("aer_outputs:", aer_outputs.shape)

            # labels: torch.Size([2, 512, 512])
            # aer_outputs: torch.Size([2, 13, 512, 512])

            # Loss for AER branch
            loss_aer = self.criterion_aer(aer_outputs, labels.long())

            # The CE loss for the SITS classification branch is done at 1.6m GSD
            # that combines the loss from the SR-diffusion model and the SITS
            #  segmentation branch

            losses["sr"] = hparams["loss_weights_aer_sat"][1] * (
                losses["sr"] + loss_sat
            )

            # The CE loss for the AER classification branch is done at 20cm GSD
            losses["aer"] = hparams["loss_weights_aer_sat"][0] * loss_aer

        else:
            losses, _, _ = self.model(img_hr, img_lr, img_lr_up)
        total_loss = sum(losses.values())
        return losses, total_loss

    def sample_and_test(self, sample):
        # Sample images and calculate evaluation metrics
        # Used for inference mode
        ret = {k: [] for k in self.metric_keys}
        ret["n_samples"] = 0
        img = sample["img"]
        txt = sample["txt"]  # torch.Size([4, 12, 3, 64, 64])
        mtd = sample["mtd"]  # torch.Size([4, 19, 64, 64])
        img_hr = sample["img_hr"]
        img_lr = sample["img_lr"]
        img_lr_up = sample["img_lr_up"]
        labels = sample["labels"]
        dates = sample["dates_encoding"]
        closest_idx = sample["closest_idx"]  # torch.Size([4, 2, 3, 160, 160])
        sc_img_hr = img_hr[:, :4, :, :]

        img_sr, final_loss = self.model.gaussian.sample(
            img,
            txt,
            mtd,
            sc_img_hr,
            img_lr,
            img_lr_up,
            dates=dates,
            closest_idx=closest_idx,

        )
        # during sampling, only the aer branch is used
        _, _, aer_outputs = self.model(img, img_sr, labels, dates, hparams)

        proba = torch.softmax(aer_outputs, dim=1)
        preds = torch.argmax(proba, dim=1)

        # print("preds:", preds.shape)
        # print("labels:", labels.shape)

        # preds: torch.Size([1, 512, 512])
        # labels: torch.Size([1, 512, 512])

        # Loop over batch
        for b in range(img_sr.shape[0]):
            s = self.measure.measure(
                img_sr[b][int(closest_idx[b].item()), :, :, :],  # SR image at t
                sc_img_hr[b],  # reference HR image
                img_lr[b][int(closest_idx[b].item()), :, :, :],  # LR input at t
                preds[b],
                labels[b],
            )
            ret["psnr"].append(s["psnr"])
            ret["ssim"].append(s["ssim"])
            ret["lpips"].append(s["lpips"])
            ret["mae"].append(s["mae"])
            ret["mse"].append(s["mse"])
            ret["shift_mae"].append(s["shift_mae"])
            ret["miou"].append(s["miou"])

            ret["n_samples"] += 1
        return img_sr, preds, ret, final_loss

    def build_optimizer(self, model):
        params = list(model.parameters())
        params = [p for p in params if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=hparams["lr"])
        return optimizer

    def build_scheduler(self, optimizer):
        # 1. Scheduler type
        # It uses torch.optim.lr_scheduler.MultiStepLR, which reduces the learning
        # rate (LR) by a factor (gamma) at specific training steps (called milestones)

        # 2. Milestones
        # This means the LR will drop twice:
        # First at 50% of decay_steps
        # Then at 90% of decay_steps
        # Example: if decay_steps = 100000, milestones = [50000, 90000].

        # 3. Gamma factor
        # At each milestone, the LR is multiplied by 0.1 (reduced by 10×).
        # Example: if LR starts at 0.001,
        # at step 50k → LR becomes 0.0001
        # at step 90k → LR becomes 0.00001.

        # 4. Effect
        # This creates a piecewise-constant decay schedule:
        # LR stays constant in between milestones.
        # At the milestone steps, LR suddenly drops.

        scheduler_param = {
            "milestones": [
                np.floor(hparams["decay_steps"] * 0.5),
                np.floor(hparams["decay_steps"] * 0.9),
            ],
            "gamma": 0.1,
        }
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_param)
