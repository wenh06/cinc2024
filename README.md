# CinC2024

[![docker-ci-and-test](https://github.com/wenh06/cinc2024/actions/workflows/docker-test.yml/badge.svg?branch=docker-test)](https://github.com/wenh06/cinc2024/actions/workflows/docker-test.yml)
[![format-check](https://github.com/wenh06/cinc2024/actions/workflows/check-formatting.yml/badge.svg)](https://github.com/wenh06/cinc2024/actions/workflows/check-formatting.yml)

Digitization and Classification of ECG Images: The George B. Moody PhysioNet Challenge 2024

[Challenge Website](https://moody-challenge.physionet.org/2024/)

<!-- toc -->

- [The Conference](#the-conference)
- [Possible solutions for the digitization task](#possible-solutions-for-the-digitization-task)

<!-- tocstop -->

## The Conference

[Conference Website](https://cinc2024.org/) |
[Unofficial Phase Leaderboard](https://docs.google.com/spreadsheets/d/e/2PACX-1vR2GLKHdS9W4Z_AOtaY_YkQrX-rY24BqQ8PmLTJW-50D9FRE-Fvijf2Gp6f3FwTN5FWx7tPb7nGEGA6/pubhtml?gid=1803759927&single=false&widget=true&headers=false)

## Possible solutions for the digitization task

- **End-to-end model**: A single model that takes the input image and produces the digitized ECG signal directly.

- **Several-stage solution**: A multi-stage solution that consists of several models, possibly including:

  - **OCR model**: Recognizes the ECG signal names and its locations in the input image, as well as other metadata.

  - **Object detection model**: Detects the area (bounding box) of the ECG signal in the input image.
    This bounding box, together with the location of the ECG signal names, can be used to crop each channel of the ECG signal.

  - **Edge sharpening algorithm**: Enhances and extracts the grid lines and the ECG signal from the cropped patches of the input image.

The end-to-end model is simpler in terms of implementation, but it may be harder to train and optimize.
Its effectiveness can not be guaranteed.

The several-stage solution may be easier to train and optimize.
But it requires more effort to design and implement the models and algorithms. (Actually a system of models and algorithms.)
