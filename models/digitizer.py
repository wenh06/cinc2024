"""
Paper ECG digitizer. Typically a segmentation model (e.g., U-Net).

Adapted from https://github.com/milesial/Pytorch-UNet/tree/master/unet
"""

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers  # noqa: F401
from PIL import Image
from torch_ecg.cfg import CFG
from torch_ecg.models.loss import MaskedBCEWithLogitsLoss
from torch_ecg.utils.download import url_is_reachable
from torch_ecg.utils.utils_nn import CkptMixin, SizeMixin

from cfg import ModelCfg
from const import INPUT_IMAGE_TYPES, MODEL_CACHE_DIR
from outputs import CINC2024Outputs
from utils.misc import get_target_sizes

# workaround for using huggingface hub in China
if os.environ.get("HF_ENDPOINT", None) is not None and (not url_is_reachable(os.environ["HF_ENDPOINT"])):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
elif os.environ.get("HF_ENDPOINT", None) is None and (not url_is_reachable("https://huggingface.co")):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODEL_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(MODEL_CACHE_DIR)
os.environ["HF_HOME"] = str(Path(MODEL_CACHE_DIR).parent)


class ECGWaveformDigitizer(nn.Module, SizeMixin, CkptMixin):
    """Paper ECG digitizer. Typically a segmentation model (e.g., U-Net).

    Parameters
    ----------
    config : CFG, optional
        The configuration of the model.
    **kwargs : Any
        Configurations that overwrite items in `config`.

    """

    __name__ = "ECGWaveformDigitizer"

    def __init__(self, config: Optional[CFG] = None, **kwargs: Any):
        super().__init__()
        self.__config = deepcopy(ModelCfg.digitizer)
        if config is not None:
            if "digitizer" in config:
                self.__config.update(deepcopy(config["digitizer"]))
            else:
                self.__config.update(deepcopy(config))
        self.__config.update(kwargs)

        self.__in_channels = {
            "raw": 3,
            "thresholded": 1,
            "both": 4,
        }[self.config.input_mode]

        if self.config.source == "custom":
            if self.config.model_name == "unet":
                self.digitizer = UNet(self.in_channels, self.config.num_classes, self.config.get("bilinear", False))
                self.criteria = MaskedBCEWithLogitsLoss()
            else:
                raise NotImplementedError(f"model_name={self.config.model_name} is not supported")
        else:
            raise NotImplementedError(f"source={self.config.source} is not supported")

        # preprocessor
        self.preprocessor = A.Compose(
            [
                A.Resize(**self.config.input_shape),
            ]
        )

    def get_input_tensors(
        self, img: INPUT_IMAGE_TYPES, labels: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, torch.Tensor]:
        """Get the input tensors from the input image and labels.

        Parameters
        ----------
        img : INPUT_IMAGE_TYPES
            The input image.
        labels : Optional[Dict[str, np.ndarray]], optional
            The mask labels.

        Returns
        -------
        Dict[str, torch.Tensor]
            The input tensors.

        """
        if self.config.source == "custom":
            if isinstance(img, torch.Tensor):
                if img.ndim == 3:
                    img = img.unsqueeze(0)
                # (B, C, H, W) -> (B, H, W, C)
                img = img.numpy().transpose(0, 2, 3, 1)
                if img.max() <= 1:
                    img = img * 255.0
                img = img.astype(np.uint8)

            if isinstance(img, (list, tuple)):
                img = [np.asarray(item) if isinstance(item, Image.Image) else item.astype(np.uint8) for item in img]

            if isinstance(img, np.ndarray):
                if img.ndim == 3:
                    img = img[np.newaxis, ...]
                if img.max() <= 1:
                    img = img * 255.0
                img = img.astype(np.uint8)

            # apply preprocessor
            if labels is None:
                preprocessed = [self.preprocessor(image=item) for item in img]
            else:  # "mask" in labels
                preprocessed = [self.preprocessor(image=item, mask=labels["mask"][idx]) for idx, item in enumerate(img)]

            if self.config.input_mode != "raw":
                # if self.training:
                #     percentile = np.random.choice([0.7, 1.0, 1.5, 2.0])
                # else:
                #     percentile = self.config.threshold_percentile
                # extra_dim = []
                # for idx in range(len(preprocessed)):
                #     threshold = np.percentile(preprocessed[idx]["image"], percentile)
                #     extra_dim.append(Image.fromarray(preprocessed[idx]["image"]).convert("L").point(lambda p: p < threshold))
                raise NotImplementedError("mode 'thresholded' and 'both' are not implemented yet")

            out_dict = {
                "image": torch.stack(
                    [torch.from_numpy(item["image"]).permute(2, 0, 1).float() / 255.0 for item in preprocessed]
                ).to(self.device)
            }
            if labels is not None:  # mask
                out_dict["mask"] = torch.stack([torch.from_numpy(item["mask"]).float() for item in preprocessed]).to(
                    self.device
                )
                out_dict["weight_mask"] = torch.stack(
                    [
                        torch.from_numpy(get_weight_mask(item["mask"], self.config.highest_weight)).float()
                        for item in preprocessed
                    ]
                ).to(self.device)

            del preprocessed

            return out_dict

    def forward(
        self,
        img: torch.Tensor,
        labels: Optional[Union[List[Dict[str, torch.Tensor]], Dict[str, list]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.

        Typically, one can get the input of this function by calling the `get_input_tensors` method.

        Parameters
        ----------
        img : torch.Tensor
            The input image tensor.
        labels : List[Dict[str, torch.Tensor]] or Dict[str, list], optional
            The mask labels.

        Returns
        -------
        Dict[str, torch.Tensor]
            The output tensors.

        """
        if self.config.source == "custom":
            logits = self.digitizer(img).squeeze(1)  # (B, C, H, W) -> (B, H, W)
            if labels is not None:
                loss = self.criteria(logits, labels["mask"], labels["weight_mask"])
            else:
                loss = None
            return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def inference(self, img: INPUT_IMAGE_TYPES, thr: float = 0.5, show: bool = False) -> CINC2024Outputs:
        """Inference.

        Parameters
        ----------
        img : INPUT_IMAGE_TYPES
            The input image.
        thr : float, default 0.5
            The threshold for binarization.
        show : bool, default False
            Whether to show the result.
            If the input image is a list of images, only the first one will be shown.

        Returns
        -------
        CINC2024Outputs
            The outputs.

        """
        original_mode = self.training
        self.eval()
        output = self.forward(self.get_input_tensors(img)["image"])
        self.train(original_mode)
        target_sizes = get_target_sizes(img)
        output["mask"] = output["logits"] > thr
        # resample the mask to the original size
        output["mask"] = [
            F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=target_size, mode="nearest")
            .squeeze(0)
            .squeeze(0)
            .cpu()
            .numpy()
            for mask, target_size in zip(output["mask"], target_sizes)
        ]
        if show:
            import matplotlib.pyplot as plt
            from PIL import Image

            if isinstance(img, (list, tuple)):
                img = img[0]
            if isinstance(img, torch.Tensor):
                img = img.cpu().detach().numpy()
            if isinstance(img, np.ndarray):
                if img.ndim == 4:
                    img = img[0]
                if img.shape[0] == 3:
                    img = np.moveaxis(img, 0, -1)

            img = Image.fromarray(img)
            img_width, img_height = img.size
            overlay_color = (0, 255, 0)
            overlay = Image.new("RGBA", img.size, overlay_color + (128,))
            overlay = Image.fromarray(np.asarray(overlay) * output["mask"][0][:, :, np.newaxis].astype(np.uint8))
            img = Image.alpha_composite(img.convert("RGBA"), overlay)

            if img_width < img_height:
                figsize = (10, 10 * img_height / img_width)
            else:
                figsize = (10 * img_width / img_height, 10)
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(img)
            plt.show()

        return CINC2024Outputs(waveform_mask=output["mask"])

    @property
    def config(self) -> CFG:
        return self.__config

    @property
    def in_channels(self) -> int:
        return self.__in_channels


class DoubleConv(nn.Sequential, SizeMixin):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None) -> None:
        if not mid_channels:
            mid_channels = out_channels
        super().__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class Down(nn.Sequential, SizeMixin):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )


class Up(nn.Module, SizeMixin):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Sequential, SizeMixin):
    """Output convolution"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size=1))


class UNet(nn.Module, SizeMixin):
    """Full assembly of the U-Net architecture."""

    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = False) -> None:
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


def get_weight_mask(mask: np.ndarray, highest_weight: int, weight_step: int = 2) -> np.ndarray:
    """Get the weight mask for the loss function.

    Parameters
    ----------
    mask : np.ndarray
        The binary mask (with 0, 1 values).
    highest_weight : int
        The highest weight added to the mask.

    Returns
    -------
    np.ndarray
        The weight mask.

    """
    weight_mask = np.ones_like(mask, dtype=np.float32) + weight_step * mask.copy().astype(np.float32)
    kernel = np.ones((5, 5), np.float32)
    for idx in range(weight_step, highest_weight, weight_step):
        weight_mask += cv2.dilate(mask, kernel, iterations=idx) * weight_step
    return weight_mask
