"""
CINC2024 models

It is a multi-head model for CINC2024. The backbone is a pre-trained image model, e.g., ResNet, DenseNet, etc.
"""

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import torch
import torch.nn as nn
from torch_ecg.cfg import CFG, DEFAULTS
from torch_ecg.utils.download import http_get, url_is_reachable
from torch_ecg.utils.misc import add_docstring
from torch_ecg.utils.utils_nn import CkptMixin, SizeMixin

from cfg import ModelCfg
from const import INPUT_IMAGE_TYPES, MODEL_CACHE_DIR
from outputs import CINC2024Outputs

from .backbone import ImageBackbone
from .detector import ECGWaveformDetector
from .digitizer import ECGWaveformDigitizer
from .heads import ClassificationHead, DigitizationHead
from .loss import get_loss_func

__all__ = [
    "MultiHead_CINC2024",
    "ImageBackbone",
    "ClassificationHead",
    "DigitizationHead",
    "get_loss_func",
    "ECGWaveformDetector",
    "ECGWaveformDigitizer",
]


# workaround for using huggingface hub in China
if os.environ.get("HF_ENDPOINT", None) is not None and (not url_is_reachable(os.environ["HF_ENDPOINT"])):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
elif os.environ.get("HF_ENDPOINT", None) is None and (not url_is_reachable("https://huggingface.co")):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODEL_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(MODEL_CACHE_DIR)
os.environ["HF_HOME"] = str(Path(MODEL_CACHE_DIR).parent)


class MultiHead_CINC2024(nn.Module, SizeMixin, CkptMixin):
    """Multi-head model for CINC2024.

    Parameters
    ----------
    config : dict
        Hyper-parameters, including backbone_name, etc.
        ref. the corresponding config file.

    """

    __DEBUG__ = True
    __name__ = "MultiHead_CINC2024"

    def __init__(self, config: Optional[CFG] = None, **kwargs: Any) -> None:
        super().__init__()
        self.__config = deepcopy(ModelCfg)
        if config is not None:
            self.__config.update(deepcopy(config))
        self.__config.update(kwargs)
        self.image_backbone = ImageBackbone(
            self.__config.backbone_name,
            source=self.__config.backbone_source,
            pretrained=True,
        )
        if self.config.backbone_source == "hf" and self.config.get("backbone_input_size", None) is not None:
            # modify the input size of the backbone if necessary
            self.image_backbone.set_input_size(self.config.backbone_input_size)
        if self.config.backbone_freeze:
            self.freeze_backbone(freeze=True)
        backbone_output_shape = self.image_backbone.compute_output_shape()
        if self.config.classification_head.include:
            self.__config.classification_head.backbone_name = self.config.backbone_name
            if self.config.classification_head.remote_checkpoints_name is not None:
                self.classification_head = ClassificationHead.from_remote(
                    url=self.config.classification_head.remote_checkpoints[
                        self.config.classification_head.remote_checkpoints_name
                    ],
                    model_dir=self.config.checkpoints,
                    weights_only=False,
                )
            else:
                self.classification_head = ClassificationHead(
                    inp_features=backbone_output_shape[0], config=self.config.classification_head
                )
        else:
            self.classification_head = None
        if self.config.digitization_head.include:
            self.__config.digitization_head.backbone_name = self.config.backbone_name
            if self.config.digitization_head.remote_checkpoints_name is not None:
                self.digitization_head = DigitizationHead.from_remote(
                    url=self.config.digitization_head.remote_checkpoints[self.config.digitization_head.remote_checkpoints_name],
                    model_dir=self.config.checkpoints,
                    weights_only=False,
                )
            else:
                self.digitization_head = DigitizationHead(inp_shape=backbone_output_shape, config=self.config.digitization_head)
        else:
            self.digitization_head = None
        assert (
            self.classification_head is not None or self.digitization_head is not None
        ), "At least one head should be included."

    def forward(
        self,
        img: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.

        Parameters
        ----------
        img : torch.Tensor
            Input image tensor.
        labels : dict, optional
            Labels for training, including
            - "dx": required for training the Dx classification head.
            - "digitization": required for training the digitization head.
            - "mask": optional for training the digitization head.

        Returns
        -------
        dict
            Predictions, including "dx" classification and digitization,
            and the loss if any of the labels is provided.

        """
        features = self.image_backbone(img)
        if self.classification_head is not None:
            dx_pred = self.classification_head(features, labels)
        else:
            dx_pred = {}
        if self.digitization_head is not None:
            digitization_pred = self.digitization_head(features, labels)
        else:
            digitization_pred = {}
        total_loss = None
        if dx_pred and "loss" in dx_pred:
            total_loss = dx_pred["loss"]
        if digitization_pred and "loss" in digitization_pred:
            if total_loss is None:
                total_loss = digitization_pred["loss"]
            else:
                total_loss += digitization_pred["loss"]
        return {
            "dx_probs": dx_pred.get("probs", None),
            "dx_logits": dx_pred.get("logits", None),
            "dx_loss": dx_pred.get("loss", None),
            "digitization": digitization_pred.get("preds", None),
            "digitization_loss": digitization_pred.get("loss", None),
            "loss": total_loss,
        }

    def get_input_tensors(
        self,
        img: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Get input tensors for the model.

        Parameters
        ----------
        img : torch.Tensor
            Input image tensor.
        labels : dict, optional
            Not used, but kept for compatibility with other models.

        Returns
        -------
        Dict[str, torch.Tensor]
            Input tensors for the image backbone model.
            The key for input image tensor is "image".
            items in `labels` are unchanged.

        """
        return {"image": self.image_backbone.get_input_tensors(img), **(labels or {})}

    @add_docstring(ImageBackbone.list_backbones.__doc__)
    @staticmethod
    def list_backbones(architectures: Optional[Union[str, Sequence[str]]] = None, source: Optional[str] = None) -> List[str]:
        return ImageBackbone.list_backbones(architectures=architectures, source=source)

    @add_docstring(ImageBackbone.freeze_backbone.__doc__)
    def freeze_backbone(self, freeze: bool = True) -> None:
        self.image_backbone.freeze_backbone(freeze)

    @torch.no_grad()
    def inference(self, img: INPUT_IMAGE_TYPES, threshold: Optional[float] = None) -> CINC2024Outputs:
        """Inference on a single image or a batch of images.

        Parameters
        ----------
        img : numpy.ndarray or torch.Tensor or PIL.Image.Image, or list
            Input image.
        threshold : float, optional
            Threshold for multi-label Dx classification.
            Defaults to `self.classification_head.config.threshold`.

        Returns
        -------
        CINC2024Outputs
            Predictions, including "dx" and "digitization".

        """
        if threshold is None:
            threshold = self.config.classification_head.threshold
        original_mode = self.training
        self.eval()
        output = self.forward(self.get_input_tensors(img)["image"])
        dx_probs = output["dx_probs"]
        # dx = [[self.config.classification_head.classes[idx] for idx, prob in enumerate(dx_probs) if prob >= threshold]]
        dx = [
            [self.config.classification_head.classes[idx] for idx, item in enumerate(probs) if item > threshold]
            for probs in dx_probs
        ]
        self.train(original_mode)
        return CINC2024Outputs(
            dx=dx,
            dx_logits=output["dx_logits"],
            dx_prob=dx_probs,
            dx_classes=self.config.classification_head.classes,
            digitization=output["digitization"],
        )

    @add_docstring(inference.__doc__)
    def inference_CINC2024(self, img: INPUT_IMAGE_TYPES) -> CINC2024Outputs:
        """alias for `self.inference`"""
        return self.inference(img)

    @property
    def config(self) -> CFG:
        return self.__config

    def save(self, path: Union[str, bytes, os.PathLike], train_config: CFG) -> None:
        """Save the model to disk.

        Parameters
        ----------
        path : `path-like`
            Path to save the model.
        train_config : CFG
            Config for training the model,
            used when one restores the model.

        Returns
        -------
        None

        """
        if not self.config.backbone_freeze:
            super().save(path, train_config)
            return

        # if the backbone is frozen, we need to save the heads only
        path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        to_save = {
            "model_config": self.config,
            "train_config": train_config,
        }
        if self.config.classification_head.include:
            to_save["classification_head_state_dict"] = self.classification_head.state_dict()
        if self.config.digitization_head.include:
            to_save["digitization_head_state_dict"] = self.digitization_head.state_dict()
        torch.save(to_save, path)

    @classmethod
    def from_checkpoint(
        cls,
        path: Union[str, bytes, os.PathLike],
        device: Optional[torch.device] = None,
        weights_only: Literal[True, False, "auto"] = "auto",
    ) -> Tuple[nn.Module, dict]:
        """Load the whole model or specific heads from a checkpoint.

        Parameters
        ----------
        path : `path-like`
            Path to the checkpoint.
            If it is a directory, then this directory should contain only one checkpoint file
            (with the extension `.pth` or `.pt`).
        device : torch.device, optional
            Map location of the model parameters,
            defaults to "cuda" if available, otherwise "cpu".
        weights_only : {"auto", True, False}, default "auto"
            Whether to load only the weights of the model.

        Returns
        -------
        model : torch.nn.Module
            The model loaded from a checkpoint.
        aux_config : dict
            Auxiliary configs that are needed for data preprocessing, etc.

        """
        if Path(path).is_dir():
            candidates = list(Path(path).glob("*.pth")) + list(Path(path).glob("*.pt"))
            assert len(candidates) == 1, "The directory should contain only one checkpoint file"
            path = candidates[0]
        if weights_only == "auto":
            if hasattr(torch.serialization, "add_safe_globals"):
                weights_only = True
            else:
                weights_only = False
        _device = device or DEFAULTS.device
        ckpt = torch.load(path, map_location=_device, weights_only=weights_only)
        aux_config = ckpt.get("train_config", None) or ckpt.get("config", None)
        assert aux_config is not None, "input checkpoint has no sufficient data to recover a model"
        kwargs = dict(
            config=ckpt["model_config"],
        )
        if "classes" in aux_config:
            kwargs["classes"] = aux_config["classes"]
        if "n_leads" in aux_config:
            kwargs["n_leads"] = aux_config["n_leads"]
        model = cls(**kwargs)
        # model.load_state_dict(ckpt["model_state_dict"])
        if kwargs["config"]["backbone_freeze"]:
            loaded_heads = False
            if "classification_head_state_dict" in ckpt:
                model.classification_head.load_state_dict(ckpt["classification_head_state_dict"])
                loaded_heads = True
            if "digitization_head_state_dict" in ckpt:
                model.digitization_head.load_state_dict(ckpt["digitization_head_state_dict"])
                loaded_heads = True
            if not loaded_heads:
                raise ValueError("No heads are loaded from the checkpoint in the frozen backbone mode.")
        else:
            model.load_state_dict(ckpt["model_state_dict"])

        return model, aux_config

    @classmethod
    def from_remote_heads(
        cls,
        url: str,
        model_dir: str,
        filename: Optional[str] = None,
        device: Optional[torch.device] = None,
        weights_only: Literal[True, False, "auto"] = "auto",
    ) -> "MultiHead_CINC2024":
        """Load the model from remote heads.

        Downloading is skipped if the model is already downloaded.

        Parameters
        ----------
        url : str
            URL of the model.
        model_dir : str
            Directory to save the model.
        filename : str, optional
            Filename for downloading the model.
        device : torch.device, optional
            Device to load the model.
        weights_only : {"auto", True, False}, default "auto"
            Whether to load only the weights of the model.

        Returns
        -------
        MultiHead_CINC2024
            The model with heads loaded from remote.

        """
        # skip downloading if the model is already downloaded
        if Path(model_dir).resolve().exists():
            candidates = (
                list(Path(model_dir).glob("*.pt"))
                + list(Path(model_dir).glob("*.pth"))
                + list(Path(model_dir).glob("*.pth.tar"))
            )
            if len(candidates) == 1:
                print(f"Loading model from local cache: {candidates[0]}")
                model_path = candidates[0].resolve()
        else:
            model_path = http_get(url, model_dir, extract=False, filename=filename)
            if Path(model_path).is_dir():
                candidates = (
                    list(Path(model_path).glob("*.pt"))
                    + list(Path(model_path).glob("*.pth"))
                    + list(Path(model_path).glob("*.pth.tar"))
                )
                assert len(candidates) == 1, "The directory should contain only one checkpoint file"
                model_path = candidates[0].resolve()
        _device = device or DEFAULTS.device
        if weights_only == "auto":
            if hasattr(torch.serialization, "add_safe_globals"):
                weights_only = True
            else:
                weights_only = False
        ckpt = torch.load(model_path, map_location=_device, weights_only=weights_only)
        aux_config = ckpt.get("train_config", None) or ckpt.get("config", None)
        assert aux_config is not None, "input checkpoint has no sufficient data to recover a model"
        kwargs = dict(
            config=ckpt["model_config"],
        )
        # backward compatibility
        if aux_config is not None:
            aux_config = {key.replace("dx_", "classification_"): value for key, value in aux_config.items()}
        kwargs["config"] = {key.replace("dx_", "classification_"): value for key, value in kwargs["config"].items()}
        ckpt["classification_head_state_dict"] = {
            key.replace("dx_", "classification_"): value for key, value in ckpt["classification_head_state_dict"].items()
        }
        if "classes" in aux_config:
            kwargs["classes"] = aux_config["classes"]
        model = cls(**kwargs)
        if model.config.classification_head.include:
            model.classification_head.load_state_dict(ckpt["classification_head_state_dict"])
        if model.config.digitization_head.include:
            model.digitization_head.load_state_dict(ckpt["digitization_head_state_dict"])
        return model
