# PIDNet Reimplementation From Open-MMLab/MMSegmentation

This repository contains an implementation of PIDNet using the mmsegmentation framework by Open-MMLab. PIDNet is a highly efficient and accurate network for real-time semantic segmentation tasks, particularly tailored for autonomous vehicle applications.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Real-Time Inference](#real-time-inference)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Overview

This implementation leverages the mmsegmentation framework to provide a flexible and powerful setup for training and evaluating PIDNet models. The repository includes configurations, training scripts, and inference tools.

## Installation

To install the required dependencies, follow these steps:

1. Clone the repository:
    ```bash
    https://github.com/anhphan2705/mmseg_pidnet.git
    cd mmseg_pidnet-main
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    If you have any problem regarding the mmcv version mismatch with PyTorch, please refer to [MMCV Installation Guide](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)

## Dataset Preparation

Prepare your dataset as per the [mmsegmentation requirements](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md).

## Training

To train the PIDNet model, use the training script with the desired configuration file:

```bash
python tools/train.py configs/pidnet/choose_a_config.py
```

Make sure to adjust the configuration file to match your dataset and training preferences.

## Real-Time Inference

To perform real-time inference using the `real_time_inference.py` script for both video and image directories, follow these steps:

1. Ensure that you have the necessary model configuration and checkpoint files.

2. Run the `real_time_inference.py` script with the appropriate arguments:

### For Video Inference

To perform real-time segmentation on a video file, use the following command:

```bash
python real_time_inference.py --video path/to/video.mp4 --config path/to/config.py --checkpoint path/to/checkpoint.pth --device cuda:0 --show
```

### For Image Directory Inference

To perform segmentation on a directory of images, use the following command:

```bash
python real_time_inference.py --images path/to/image_directory/* --config path/to/config.py --checkpoint path/to/checkpoint.pth --device cuda:0 --show
```

### Arguments

- `--video`: Path to the video file for inference.
- `--images`: Path to the directory containing images for inference.
- `--config`: Path to the model configuration file.
- `--checkpoint`: Path to the model checkpoint file.
- `--device`: Device to be used for inference (`cpu` or `cuda:0`).
- `--out`: Path to the output directory for images or video file path for saving results.
- `--show`: If specified, display the video or images during processing.
- `--wait-time`: Interval of show in seconds, default is 0.001 seconds.

This script will load the trained model, perform segmentation on the input video or images, and display or save the results based on the provided arguments.

## Acknowledgements

This implementation is heavily based on the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) repository by Open-MMLab.