# Models for the CINC 2024 Challenge

This folder (module) contains the models for the CINC 2024 Challenge. The models are decomposed mainly into two parts:

- **Pre-trained image backbone**: This part is responsible for extracting features from the input images. The backbone is pre-trained on the ImageNet dataset and is frozen during training.

- **Dx head**: This part is responsible for making classification predictions ("Normal" or "Abnormal") based on the features extracted by the image backbone. The Dx head is trained (fine-tuned) on the CINC 2024 dataset.

- **Digitization head**: This part is responsible for recovering the digitized ECG signal from the input image. The digitization head is trained (fine-tuned) on the CINC 2024 dataset.

## Pre-trained image backbone

There are typically 3 sources of pre-trained image backbones:

- **Huggingface Transformers**

- **PyTorch-Image-Models (timm)**

- **Torchvision**

**Huggingface Transformers** is recommended for the following reasons:

    - It has `AutoBackbone` class which produces feature maps from images,
        without the need to handle extra pooling layers and classifier layers.
        The rest two sources do not have a method for creating feature map extractors directly,
        and the models does not in general have a common method for extracting feature maps (e.g. calling methods like `forward_features`).

    - It has `AutoImageProcessor` class which can be used to preprocess images before feeding them to the model,
        so that one does not need to manually preprocess the images before feeding them to the model.
        The rest two sources do not have a method for creating image preprocessors directly (`timm` is better).
        One has to use for example `IMAGENET_DEFAULT_MEAN` and `IMAGENET_DEFAULT_STD` to normalize the images manually.

## Dx head

Dx head is typically a multi-layer perceptron (MLP) that takes the features extracted by the image backbone and produces classification predictions.

## Digitization head

Digitization head can be considered as a generative model that takes the features extracted by the image backbone and produces the digitized ECG signal.
A `max_len` parameter is used to determine the maximum length of the digitized ECG signal.

The digitization head typically consists of a few fully connected layers, followed by a softmax layer, like the Dx head.
The difference is that to generate the digitized values,
the image features (of shape ``(batch_size, num_features, height, width)``) are mapped to (via some layers)
feature tensors of shape ``(batch_size, num_leads, max_len)``, where ``num_leads`` is the number of ECG leads,
and ``max_len`` is the maximum length of the ECG signals.
This is done by the following steps:

1. The image features are flattened to get a feature vector of shape ``(batch_size, num_features * height * width)``.

2. The feature vector is fed into a few fully connected layers to get a feature tensor of shape ``(batch_size, num_leads * max_len)``.

3. The feature tensor is reshaped to get the final feature tensor of shape ``(batch_size, num_leads, max_len)``.

## Loss functions for digitization head

- SNR loss: Signal-to-noise ratio loss.
  The signal-to-noise ratio (SNR) is defined as the ratio of the power of the signal to the power of the noise.
  The SNR loss is defined as the negative SNR.

- DTW loss: Dynamic time warping loss.
  DTW is a method for measuring similarity between two temporal sequences that may vary in speed.

- MAE loss: Mean absolute error loss.

- RMSE loss: Root mean squared error loss.

- KS loss: Kolmogorov-Smirnov loss, inspired by the Kolmogorov-Smirnov test.
  KS test is a non-parametric test of the equality of continuous, one-dimensional probability distributions
  that can be used to test whether a sample came from a given reference probability distribution (one-sample KS test),
  or to test whether two samples came from the same distribution (two-sample KS test).

- ASCI loss: Adaptive signed correlation index loss. It is a measure to quantify the morphological similarity between signals.
