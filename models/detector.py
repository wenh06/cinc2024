"""
Waveform detector model, which detects the bounding boxes of the waveforms in the ECG images.
"""

import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
import torch.nn as nn
import transformers
from torch_ecg.cfg import CFG
from torch_ecg.utils.misc import CitationMixin, list_sum
from torch_ecg.utils.utils_nn import CkptMixin, SizeMixin

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


class ECGWaveformDetector(nn.Module, CitationMixin, SizeMixin, CkptMixin):
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

    def __init__(self, config: Optional[CFG] = None, **kwargs: Any) -> None:
        super().__init__()
        self.__config = deepcopy(ModelCfg.object_detection)
        if config is not None:
            self.__config.update(deepcopy(config))
        self.__config.update(kwargs)
        if self.config.source.lower() == "hf":
            # The preprocessor typically expects the annotations to be in the following format:
            #  {'image_id': int, 'annotations': List[Dict]}, where each dictionary is a COCO object annotation
            self.preprocessor = transformers.AutoImageProcessor.from_pretrained(
                self.config.model_name,
                cache_dir=MODEL_CACHE_DIR,
            )
            self.augmentor = None
            self.detector = transformers.AutoModelForObjectDetection.from_pretrained(
                self.config.model_name,
                label2id=self.config["label2id"],
                id2label={v: k for k, v in self.config["label2id"].items()},
                cache_dir=MODEL_CACHE_DIR,
                ignore_mismatched_sizes=True,
            )
        elif self.config.source == "mm":
            raise NotImplementedError
        elif self.config.source == "de":
            raise NotImplementedError
        else:
            raise ValueError(f"Unsupported source: {self.config.source}")

    def get_input_tensors(
        self, img: INPUT_IMAGE_TYPES, labels: Optional[Union[Dict[str, list], List[Dict[str, torch.Tensor]]]] = None
    ) -> Union[Dict[str, torch.Tensor], transformers.BatchFeature]:
        """Get input tensors for the model.

        Parameters
        ----------
        img : numpy.ndarray, or torch.Tensor, or PIL.Image.Image, or list or tuple
            Input image(s).
        labels : Dict[str, list] or List[Dict[str, torch.Tensor]], optional
            The bounding box labels.
            If the source is "hf", the label dictionary should have the following keys:
            - "image_id" (`int`): The image id.
            - "bbox" (`List[Dict]`): List of bounding boxes for an image.
              The bounding boxes are typically of COCO format (https://cocodataset.org/#format-data):
              Each bounding boxes annotation should be a dictionary.
              An image can have no annotations, in which case the list should be empty.
              The bounding boxes annotation dictionary can have the following keys:
              - "annotations" (`List[Dict]`, required): List of annotations for an image.
                Each annotation is a dictionary with the following keys:
                - "bbox" (`List[float]`, required): The bounding box of the object, of the form
                  [top left x, top left y, width, height]
                - "category_id" (`int`, required): The category id, the same as the category id in the label2id mapping.
                - "iscrowd" (`int`, optional): The iscrowd flag. NOT used in this project.
                - "area" (`float`, required): The area of the object, can be pre-calculated as width * height.
              - "iamge_id" (`int`): The image id.
            If is a list of dictionaries, if should have the same structure as the "bbox" key
            in the dictionary mentioned above.

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
        if self.config.source == "hf":
            if labels is not None:
                if "annotations" in labels:
                    output = self.preprocessor(images=img, annotations=labels, return_tensors="pt")
                else:
                    output = self.preprocessor(images=img, annotations=labels["bbox"], return_tensors="pt")
                output.pop("pixel_mask")
                output = self.move_to_model_device(output)
                # rename "pixel_values" to "image"
                output["image"] = output.pop("pixel_values")
            else:
                output = {"image": self.preprocessor(img).convert_to_tensors("pt")["pixel_values"].to(self.device)}
        elif self.config.source == "timm":
            # not tested
            if isinstance(img, (np.ndarray, torch.Tensor)):
                img_ndim = img.ndim
            elif isinstance(img, (PIL.Image.Image)):
                img_ndim = 3
            elif isinstance(img, (list, tuple)):
                img_ndim = 4
            else:
                raise ValueError(f"Input tensor has invalid type: {type(img)}")
            if img_ndim == 3:
                output = self.preprocessor(img).to(self.device)
            elif img_ndim == 4:
                output = torch.stack([self.preprocessor(item).to(self.device) for item in img])
            else:
                raise ValueError(f"Input tensor has invalid shape: {img.shape}")
            raise NotImplementedError
        elif self.config.source == "tv":
            output = self.preprocessor(img).to(self.device)
            raise NotImplementedError
        return output

    def forward(
        self,
        img: torch.Tensor,
        labels: Optional[Union[List[Dict[str, torch.Tensor]], Dict[str, list], transformers.BatchFeature]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.

        Typically, one can get the input of this function by calling the `get_input_tensors` method.

        Parameters
        ----------
        img : torch.Tensor
            The input image tensor.
        labels : List[Dict[str, torch.Tensor]] or Dict[str, list] or transformers.BatchFeature, optional
            The bbox labels.
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.
            If is a dictionary or `BatchFeature`,
            it should have a "labels" key, whose value is a list of dictionaries mentioned above.

        Returns
        -------
        Dict[str, torch.Tensor]
            The output of the model.

        """
        if self.config.source == "hf":
            if isinstance(labels, (transformers.BatchFeature, dict)):
                labels = labels["labels"]
            outputs = self.detector(img, labels=labels)
        elif self.config.source == "mm":
            raise NotImplementedError
        elif self.config.source == "de":
            raise NotImplementedError
        return outputs

    @torch.no_grad()
    def inference(self, img: INPUT_IMAGE_TYPES, threshold: Optional[float] = None) -> CINC2024Outputs:
        """Inference on a single image or a batch of images.

        Parameters
        ----------
        img : numpy.ndarray, or torch.Tensor, or PIL.Image.Image, or list or tuple
            Input image.
        threshold : float, optional
            The threshold for filtering the bounding boxes.
            If None, the threshold is set to the value of `self.config.bbox_thr`.

        Returns
        -------
        CINC2024Outputs
            Predictions, including "bbox".

        """
        original_mode = self.training
        self.eval()
        output = self.forward(self.get_input_tensors(img)["image"])
        self.train(original_mode)
        target_sizes = get_target_sizes(img)
        # outputs converted to a list of dictionaries with keys: "scores", "labels", "boxes"
        threshold = threshold if threshold is not None else self.config.bbox_thr
        output = self.preprocessor.post_process_object_detection(output, threshold=threshold, target_sizes=target_sizes)
        # dictionary values to numpy array
        for idx, list_item in enumerate(output):
            for key in list_item:
                if isinstance(list_item[key], torch.Tensor):
                    list_item[key] = list_item[key].cpu().detach().numpy()
            list_item["boxes"] = np.round(list_item["boxes"]).astype(int)
            list_item["image_size"] = target_sizes[idx]
            list_item["category_id"] = list_item.pop("labels")
            list_item["category_name"] = [self.detector.config.id2label[cl] for cl in list_item["category_id"]]
        return CINC2024Outputs(bbox=output)

    def move_to_model_device(
        self, input_tensors: Union[torch.Tensor, transformers.BatchFeature, list, dict]
    ) -> Union[torch.Tensor, transformers.BatchFeature, list, dict]:
        """Move the input tensors to the device of the model.

        Parameters
        ----------
        input_tensors : torch.Tensor or transformers.BatchFeature or list or dict
            The input tensors.

        Returns
        -------
        torch.Tensor or transformers.BatchFeature or list or dict
            The input tensors moved to the device of the model.

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
        return input_tensors

    @property
    def config(self) -> CFG:
        return self.__config


def get_target_sizes(img: INPUT_IMAGE_TYPES, channels: int = 3) -> List[Tuple[int, int]]:
    """Get the target sizes of the input image(s).

    Parameters
    ----------
    img : numpy.ndarray, or torch.Tensor, or PIL.Image.Image, or list or tuple
        Input image.
    channels : int, default 3
        The number of channels of the input image.
        Used to determine the channel dimension of the input image.

    Returns
    -------
    List[Tuple[int, int]]
        The list containing the target size `(height, width)` of each image.

    """
    if isinstance(img, (list, tuple)):
        target_sizes = list_sum(get_target_sizes(item, channels) for item in img)
    elif isinstance(img, (np.ndarray, torch.Tensor)):
        if img.ndim == 3:
            if img.shape[0] == channels:  # channels first
                target_sizes = [tuple(img.shape[1:])]
            else:  # channels last
                target_sizes = [tuple(img.shape[:-1])]
        elif img.ndim == 4:
            target_sizes = list_sum(get_target_sizes(item, channels) for item in img)
    elif isinstance(img, PIL.Image.Image):
        target_sizes = [img.size[::-1]]
    return target_sizes
