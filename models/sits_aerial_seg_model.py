import os
import torch
import timm
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from utils.hparams import hparams
from timm.layers import create_conv2d
from models.fusion_module.aer_cross_sat_atts import FFCA
from models.decoders.unet_former_decoder import UNetFormerDecoder
from models.encoders.t_convformer import TConvFormer
from models.decoders.unet_decoder import UNetDecoder


class SITSAerialSegmenter(nn.Module):
    def __init__(self, gaussian, config):
        super().__init__()
        self.gaussian = gaussian
        self.config = config
        self.embed_dim = config["embed_dim"]
        self.decoder_channels = config["decoder_channels"]
        self.num_classes = config["num_classes"]
        self.dropout = config["dropout"]
        self.window_size = config["window_size"]

        # 1. SITS Network
        self.sr_sits_enc = TConvFormer(
            input_size=(config["sr_patch_size"], config["sr_patch_size"]),
            stem_channels=64,
            block_channels=config["block_channels"],  # [64, 128, 256, 512]
            block_layers=config["block_layers"],  # [2, 2, 5, 2]
            head_dim=32,
            stochastic_depth_prob=0.2,
            partition_size=4,
        )

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
            model_dict = self.sr_sits_enc.state_dict()
            old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
            model_dict.update(old_dict)
            print(f"| load 'cond model' from '{weights_path}'.")
            self.sr_sits_enc.load_state_dict(model_dict)

        # 2. Aerial Encoder
        self.aer_net_enc = timm.create_model(
            "maxvit_tiny_tf_512.in1k",
            pretrained=True,
            features_only=True,
            num_classes=config["num_classes"],
        )

        # Get first conv layer (usually called 'stem.conv' in MaxViT)
        conv1 = (
            self.aer_net_enc.stem.conv1
        )  # <-- sometimes it's model.stem.conv or model.conv_stem, check print(model)

        # Create new conv with 5 input channels instead of 3
        new_conv = create_conv2d(
            in_channels=config["num_channels_aer"],  # Use num_channels from config
            out_channels=conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=1,  # original padding was None, but we set it to 1 for compatibility
            bias=conv1.bias is not None,
        )

        # Initialize the first 3 channels with pretrained weights
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = conv1.weight  # copy RGB weights
            # Initialize the extra channels randomly (e.g., Kaiming normal)
            nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :])

        # Replace the old conv with the new one
        self.aer_net_enc.stem.conv1 = new_conv

        # give latent_diff access to aer_net_enc
        self.gaussian.aer_net_enc = self.aer_net_enc

        encoder_channels = [
            config["embed_dim"],  # 64
            config["embed_dim"] * 2,  # 128
            config["embed_dim"] * 4,  # 256
            config["embed_dim"] * 8,  # 512
        ]
        self.sr_sits_dec = UNetDecoder(
            encoder_channels,  #  remove the last channels dim.
            config["decoder_channels"],
            config["dropout"],
            config["window_size"],
            config["num_classes"],
        )

        # 3. Aerial Decoder from U-Net Former paper
        self.aer_net_dec = UNetFormerDecoder(
            encoder_channels,
            config["decoder_channels"],
            config["dropout"],
            config["window_size"],
            config["num_classes"],
        )
        # 4. Fusion Module
        self.fusion_module = FFCA(
            aer_channels_list=[128, 256, 512],
            sits_channels_list=[64, 128, 256],
            num_heads=8,
        )

    def forward(
        self,
        aerial: torch.FloatTensor,
        img_sr: torch.FloatTensor,
        labels: torch.FloatTensor,
        dates: torch.FloatTensor,
        config,
    ):
        # aerial:  torch.Size([4, 5, 512, 512])
        h_hr, w_hr = aerial.size()[-2:]
        h_sr, w_sr = img_sr.size()[-2:]

        # SR-SITS branch

        ## Encoder
        red_temp_feats, _ = self.sr_sits_enc(img_sr, dates)
        ## Decoder (USE ONLY DURING TRAINING)
        sits_logits, multi_lvls_cls = self.sr_sits_dec(red_temp_feats, h_sr, w_sr)

        # Aerial branch
        hr_0, hr_1, hr_2, hr_3, hr_4 = self.aer_net_enc(aerial)

        # print("---------------SR Reduced Temp Feats-------------------")
        # print("red_temp_feats 0:", red_temp_feats[0].shape)
        # print("red_temp_feats 1:", red_temp_feats[1].shape)
        # print("red_temp_feats 2:", red_temp_feats[2].shape)

        # # red_temp_feats 0: torch.Size([2, 64, 64, 64])
        # # red_temp_feats 1: torch.Size([2, 128, 32, 32])
        # # red_temp_feats 2: torch.Size([2, 256, 16, 16])
        # print()

        # print("--------------- Multi res SR Feats-------------------")
        # print("multi_lvls_cls 0:", multi_lvls_cls[0].shape)
        # print("multi_lvls_cls 1:", multi_lvls_cls[1].shape)
        # print("multi_lvls_cls 2:", multi_lvls_cls[2].shape)

        # # multi_lvls_cls 0: torch.Size([2, 13, 64, 64])
        # # multi_lvls_cls 1: torch.Size([2, 13, 64, 64])
        # # multi_lvls_cls 2: torch.Size([2, 13, 64, 64])

        # Fusion FFCA
        fus_2, fus_3, fus_4 = self.fusion_module([hr_2, hr_3, hr_4], red_temp_feats)

        # print("---------------SR Reduced Temp Feats-------------------")
        # print("fusion outputs 2:", fus_2.shape)
        # print("fusion outputs 3:", fus_3.shape)
        # print("fusion outputs 4:", fus_4.shape)

        # # fusion outputs 2: torch.Size([2, 128, 64, 64])
        # # fusion outputs 3: torch.Size([2, 256, 32, 32])
        # # fusion outputs 4: torch.Size([2, 512, 16, 16])
        # print()

        # Decoder
        logits = self.aer_net_dec(hr_0, hr_1, fus_2, fus_3, fus_4, h_hr, w_hr)

        return sits_logits, multi_lvls_cls, logits
