"""
Waveform detector model, which detects the bounding boxes of the waveforms in the ECG images.
"""

import os
from copy import deepcopy
from typing import Dict, List, Optional, Union

import numpy as np
import PIL
import torch
import transformers
from torch_ecg.cfg import CFG
from torch_ecg.utils.misc import CitationMixin
from torch_ecg.utils.utils_nn import SizeMixin

from cfg import ModelCfg
from const import INPUT_IMAGE_TYPES, MODEL_CACHE_DIR
from outputs import CINC2024Outputs
from utils.misc import url_is_reachable

# workaround for using huggingface hub in China
if os.environ.get("HF_ENDPOINT", None) is not None and (not url_is_reachable(os.environ["HF_ENDPOINT"])):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
elif os.environ.get("HF_ENDPOINT", None) is None and (not url_is_reachable("https://huggingface.co")):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)


class ECGWaveformDetector(CitationMixin, SizeMixin):
    """Waveform detector model, which detects the bounding boxes of the waveforms in the ECG images.

    Parameters
    ----------
    name_or_path: Union[str, bytes, os.PathLike]
        The name or path of the model.
        e.g. "facebook/detr-resnet-50", "microsoft/conditional-detr-resnet-50", "hustvl/yolos-small",
        or "/path/to/model".
    source: {"hf", "mm"}, default "hf"
        The source of the model.
        It can be "hf" (Hugging Face) or "mm" (MM-Detection) or "de" (Detectron2).
    pretrained: bool, default True
        Whether to load the pretrained model weights.

    References
    ----------
    .. [1] https://huggingface.co/docs/transformers/en/tasks/object_detection

    """

    def __init__(
        self,
        name_or_path: Union[str, bytes, os.PathLike],
        source: str = "hf",
        pretrained: bool = True,
        config: Optional[CFG] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.__config = deepcopy(ModelCfg.object_detection.copy())
        if config is not None:
            self.__config.update(deepcopy(config))
        self.name_or_path = name_or_path
        self.source = source.lower()
        if self.source == "hf":
            self.preprocessor = transformers.AutoImageProcessor.from_pretrained(
                name_or_path,
                cache_dir=MODEL_CACHE_DIR,
            )
            self.augmentor = None
            self.detector = transformers.AutoModelForObjectDetection.from_pretrained(
                name_or_path,
                label2id=self.config.label2id,
                id2label={v: k for k, v in self.config.label2id.items()},
                cache_dir=MODEL_CACHE_DIR,
            )
        elif self.source == "mm":
            raise NotImplementedError
        elif self.source == "de":
            raise NotImplementedError
        else:
            raise ValueError(f"Unsupported source: {source}")

    def get_input_tensors(
        self, x: INPUT_IMAGE_TYPES, labels: Optional[Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]] = None
    ) -> Union[torch.Tensor, transformers.BatchFeature]:
        """Get input tensors for the model.

        Parameters
        ----------
        x : numpy.ndarray or torch.Tensor or PIL.Image.Image or list
            Input image(s).
        labels : Dict[str, torch.Tensor] or List[Dict[str, torch.Tensor]], optional
            The bbox labels.
            If the source is "hf", the label dictionaries should have the following keys:
            - "image_id" (`int`): The image id.
            - "annotations" (`List[Dict]`): List of annotations for an image.
              The annotations are typically of COCO format (https://cocodataset.org/#format-data):
              Each annotation should be a dictionary. An image can have no annotations, in which case the list should be empty.
              The annotation dictionary can have the following keys:
                - "bbox" (`List[float]`, required): The bounding box of the object, of the form
                  [top left x, top left y, width, height]
                - "category_id" (`int`, required): The category id, the same as the category id in the label2id mapping.
                - "iscrowd" (`int`, optional): The iscrowd flag. NOT used in this project.
                - "area" (`float`, required): The area of the object, can be pre-calculated as width * height.

        Returns
        -------
        torch.Tensor or transformers.BatchFeature
            Input tensor(s) for the detector model.
            A :class:`transformers.BatchFeature` object if the source is "hf" and `labels` is not None.
            The `BatchFeature` object contains the following keys: "pixel_values", "labels", "pixel_mask" (NOT used in this project)
            "pixel_values" is a :class:`torch.Tensor` object. "labels" is a :class:`list` of :class:`dict` objects.
            A :class:`torch.Tensor` object if the source is "hf" and `labels` is None.

        """
        assert self.preprocessor is not None, "Set up the preprocessor first."
        if isinstance(x, (np.ndarray, torch.Tensor)):
            x_ndim = x.ndim
        elif isinstance(x, (PIL.Image.Image)):
            x_ndim = 3
        elif isinstance(x, (list, tuple)):
            x_ndim = 4
        else:
            raise ValueError(f"Input tensor has invalid type: {type(x)}")
        if self.source == "hf":
            if labels is not None:
                x = self.preprocessor(x, labels, return_tensors="pt")
                # move the tensors to the device of the model
                self.move_to_model_device(x)
            else:
                x = self.preprocessor(x).convert_to_tensors("pt")["pixel_values"].to(self.device)
        elif self.source == "timm":
            if x_ndim == 3:
                x = self.preprocessor(x).to(self.device)
            elif x_ndim == 4:
                x = torch.stack([self.preprocessor(img).to(self.device) for img in x])
            else:
                raise ValueError(f"Input tensor has invalid shape: {x.shape}")
            raise NotImplementedError
        elif self.source == "tv":
            x = self.preprocessor(x).to(self.device)
            raise NotImplementedError
        return x

    def forward(
        self,
        img: torch.Tensor,
        labels: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.

        Typically, one can get the input of this function by calling the `get_input_tensors` method.

        Parameters
        ----------
        img : torch.Tensor
            The input image tensor.
        labels : Dict[str, torch.Tensor], optional
            The bbox labels.
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

        Returns
        -------
        Dict[str, torch.Tensor]
            The output of the model.

        """
        if self.source == "hf":
            outputs = self.detector(img, labels=labels)
        elif self.source == "mm":
            raise NotImplementedError
        elif self.source == "de":
            raise NotImplementedError
        return outputs

    @torch.no_grad()
    def inference(self, img: INPUT_IMAGE_TYPES) -> CINC2024Outputs:
        """Inference on a single image or a batch of images.

        Parameters
        ----------
        img : numpy.ndarray or torch.Tensor or PIL.Image.Image, or list
            Input image.

        Returns
        -------
        CINC2024Outputs
            Predictions, including "boxes".

        """
        original_mode = self.training
        self.eval()
        output = self.forward(self.image_backbone.get_input_tensors(img))
        self.train(original_mode)
        # outputs converted to a list of dictionaries with keys: "scores", "labels", "boxes"
        output = self.preprocessor.post_process_object_detection(output, threshold=self.config.bbox_thr)
        # dictionary values to numpy array
        for list_item in output:
            for k in list_item:
                if isinstance(list_item[k], torch.Tensor):
                    list_item[k] = list_item[k].cpu().detach().numpy()
        return CINC2024Outputs(
            bbox={
                "xmin": output["boxes"][:, 0],
                "ymin": output["boxes"][:, 1],
                "xmax": output["boxes"][:, 2],
                "ymax": output["boxes"][:, 3],
                "class": [self.config.class_names[i] for i in output["labels"]],
                "score": output["scores"],
            }
        )

    def move_to_model_device(self, input_tensors: Union[torch.Tensor, transformers.BatchFeature, list, dict]) -> None:
        """Move the input tensors to the device of the model.

        Parameters
        ----------
        input_tensors : torch.Tensor or transformers.BatchFeature or list or dict
            The input tensors.

        Returns
        -------
        None

        """
        if isinstance(input_tensors, torch.Tensor):
            input_tensors = input_tensors.to(self.device)
        elif isinstance(input_tensors, list):
            for i, tensor in enumerate(input_tensors):
                input_tensors[i] = self.move_to_model_device(tensor)
        elif isinstance(input_tensors, (dict, transformers.BatchFeature)):
            for k in input_tensors:
                input_tensors[k] = self.move_to_model_device(input_tensors[k])
        else:
            pass  # do nothing

    @property
    def config(self) -> CFG:
        return self.__config
