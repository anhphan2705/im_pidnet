# PIDNet Reimplementation From Open-MMLab/MMSegmentation

This repository contains an implementation of PIDNet using the mmsegmentation framework by Open-MMLab. PIDNet is a highly efficient and accurate network for real-time semantic segmentation tasks, particularly tailored for autonomous vehicle applications.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
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

## Inference

For inference, refer to [mmsegmentation guide](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/3_inference.md).

Run this sample `inference.py` script to see a sample output:

```bash
python inference.py
```

This script will load the trained model and perform segmentation on the input image.

## Acknowledgements

This implementation is heavily based on the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) repository by Open-MMLab.