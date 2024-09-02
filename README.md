# CinC2024

[![docker-ci-and-test](https://github.com/wenh06/cinc2024/actions/workflows/docker-test.yml/badge.svg?branch=docker-test)](https://github.com/wenh06/cinc2024/actions/workflows/docker-test.yml)
[![format-check](https://github.com/wenh06/cinc2024/actions/workflows/check-formatting.yml/badge.svg)](https://github.com/wenh06/cinc2024/actions/workflows/check-formatting.yml)

Digitization and Classification of ECG Images: The George B. Moody PhysioNet Challenge 2024

[Challenge Website](https://moody-challenge.physionet.org/2024/)

The figure below demonstrates the framework of the proposed method in this project.

![The framework](images/framework.svg)

:skull::skull::skull: **SUPER BIG MISTAKE**: The loss function for the classification head of the official phase (multi-label) is NOT changed from the cross-entropy loss used in the unofficial phase (single-label) to the asymmetric loss. See [the commit](https://github.com/wenh06/cinc2024/commit/188e7b21d11045af36341a84105b4f9d63c1e6cc) for the details. :skull::skull::skull:

<!-- toc -->

- [The Conference](#the-conference)
- [Possible solutions for the digitization task](#possible-solutions-for-the-digitization-task)

<!-- tocstop -->

## The Conference

[Conference Website](https://cinc2024.org/) |
[Unofficial Phase Leaderboard](https://docs.google.com/spreadsheets/d/e/2PACX-1vR2GLKHdS9W4Z_AOtaY_YkQrX-rY24BqQ8PmLTJW-50D9FRE-Fvijf2Gp6f3FwTN5FWx7tPb7nGEGA6/pubhtml?gid=1803759927&single=false&widget=true&headers=false) |
[Official Phase Leaderboard](https://docs.google.com/spreadsheets/d/e/2PACX-1vRxoN5oxymRHNa5XFjautP0Jn6BqtrX8gVkoW6M3FPzEYvi8ma-7sF9-ftU8gwkX2XCcunkYbCxdO3E/pubhtml?rm=minimal&gid=1894271459&gid=2117462787&single=false&widget=true&headers=false)

:point_right: [Back to TOC](#cinc2024)

## Possible solutions for the digitization task

<details>
<summary>Click to view the details</summary>

- **End-to-end model** (NOT adopted): A single model that takes the input image and produces the digitized ECG signal directly.

- **Several-stage solution** (adopted): A multi-stage solution that consists of several models, possibly including:

  - ~~**OCR model**: Recognizes the ECG signal names and its locations in the input image, as well as other metadata.~~
    ~~For example, using [EasyOCR](https://github.com/JaidedAI/EasyOCR), or [Tesseract](https://github.com/tesseract-ocr/tesseract),~~
    ~~or [TrOCR](https://huggingface.co/docs/transformers/en/model_doc/trocr).~~

  - **Object detection model**: Detects the area (bounding box) of the ECG signal in the input image.
    This bounding box, together with the location of the ECG signal names, can be used to crop each channel of the ECG signal.

  - ~~**Edge sharpening algorithm**: Enhances and extracts the grid lines and the ECG signal from the cropped patches of the input image.~~

  - **Segmentation model**: Segments the ECG signal from the cropped patches of the input image.
    This model can be a U-Net, a DeepLabV3, or a Mask R-CNN, etc.

The end-to-end model is simpler in terms of implementation, but it may be harder to train and optimize.
Its effectiveness can not be guaranteed.

The several-stage solution may be easier to train and optimize.
But it requires more effort to design and implement the models and algorithms. (Actually a system of models and algorithms.)

</details>

:point_right: [Back to TOC](#cinc2024)

## Description of the files/folders(modules)

### Files

<details>
<summary>Click to view the details</summary>

- [README.md](README.md): this file, serves as the documentation of the project.
- [cfg.py](cfg.py): the configuration file for the whole project.
- [data_reader.py](data_reader.py): data reader, including data downloading, file listing, data loading, etc.
- [dataset.py](dataset.py): dataset class, which feeds data to the models.
- [Dockerfile](Dockerfile): docker file for building the docker image for submissions.
- [evaluate_model.py](evaluate_model.py), [helper_code.py](helper_code.py), [remove_hidden_data.py](remove_hidden_data.py), [run_model.py](run_model.py), [train_model.py](train_model.py): scripts inherited from the [official baseline](https://github.com/physionetchallenges/python-example-2024.git) and [official scoring code](https://github.com/physionetchallenges/evaluation-2024.git). Modifications on these files are invalid and are immediately overwritten after being pulled by the organizers (or the submission system).
- [sync_official.py](sync_official.py): script for synchronizing data from the official baseline and official scoring code.
- [requirements.txt](requirements.txt), [requirements-docker.txt](requirements-docker.txt), [requirements-no-torch.txt](requirements-no-torch.txt): requirements files for different purposes.
- [team_code.py](team_code.py): entry file for the submissions.
- [test_local.py](test_local.py), [test_docker.py](test_docker.py), [test_run_challenge.sh](test_run_challenge.sh): scripts for testing the docker image and the local environment. The latter 2 files along with the [docker-test action](.github/workflows/docker-test.yml) are used for CI. Passing the CI almost guarantees that the submission will run successfully in the official environment, except for potential GPU-related issues (e.g. model weights and data are on different devices, i.e. CPU and GPU, in which case torch will raise an error).
- [trainer.py](trainer.py): trainer class, which trains the models.
- [submissions](submissions): log file for the submissions, including the key hyperparameters, the scores received, commit hash, etc. The log file is updated after each submission and organized as a YAML file.

</details>

### Folders(Modules)

<details>
<summary>Click to view the details</summary>

- [official_baseline](official_baseline): the official baseline code, included as a submodule.
- [official_scoring_metric](official_scoring_metric): the official scoring code, included as a submodule.
- [ecg-image-kit](ecg-image-kit): a submodule for the ECG image processing and generating toolkit, provided by the organizers.
- [models](models): folder for model definitions, including [image backbones](models/backbone.py), [Dx head, digitization head](models/heads.py), [custom losses](models/loss.py), [waveform detector](models/waveform_detector.py), etc.
- [utils](utils): various utility functions, including a [ECG simulator](utils/ecg_simulator.py) for generating synthetic ECG signals, [ecg image generator](utils/ecg_image_generator) which is an enhanced version of the [ecg-image-kit](ecg-image-kit), etc.

</details>

:point_right: [Back to TOC](#cinc2024)
