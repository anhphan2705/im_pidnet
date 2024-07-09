# Improved PIDNet implementation From Open-MMLab/MMSegmentation


This repository contains an improved implementation of PIDNet from the mmsegmentation framework by Open-MMLab. PIDNet is a highly efficient and accurate network for real-time semantic segmentation tasks, particularly tailored for autonomous vehicle applications. The repository includes configurations, training scripts, and significantly improved inference tools.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Real-Time Inference](#real-time-inference)
- [Acknowledgements](#acknowledgements)

## Installation

To install the required dependencies, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/anhphan2705/mmseg_pidnet.git
    cd mmseg_pidnet
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

```
mmseg_pidnet
├── mmseg
├── tools
├── configs
├── samples
├── real_time_inference.py
├── model-index.yml
├── README.md
├── requirements.txt
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── coco_stuff10k
│   │   ├── images
│   │   │   ├── train2014
│   │   │   ├── test2014
│   │   ├── annotations
│   │   │   ├── train2014
│   │   │   ├── test2014
│   │   ├── imagesLists
│   │   │   ├── train.txt
│   │   │   ├── test.txt
│   │   │   ├── all.txt
│   ├── coco_stuff164k
│   │   ├── images
│   │   │   ├── train2017
│   │   │   ├── val2017
│   │   ├── annotations
│   │   │   ├── train2017
│   │   │   ├── val2017
|   ├── dark_zurich
|   │   ├── gps
|   │   │   ├── val
|   │   │   └── val_ref
|   │   ├── gt
|   │   │   └── val
|   │   ├── LICENSE.txt
|   │   ├── lists_file_names
|   │   │   ├── val_filenames.txt
|   │   │   └── val_ref_filenames.txt
|   │   ├── README.md
|   │   └── rgb_anon
|   │   │   ├── val
|   │   │   └── val_ref
|   ├── NighttimeDrivingTest
|   │   ├── gtCoarse_daytime_trainvaltest
|   │   │   └── test
|   │   │       └── night
|   │   └── leftImg8bit
|   │       └── test
|   │           └── night
│   ├── bdd100k
│   │   ├── images
│   │   │   └── 10k
│   │   │       ├── test
│   │   │       ├── train
│   │           └── val
│   │   └── labels
│   │       └── sem_seg
│   │           ├── colormaps
│   │           │   ├──train
│   │           │   └──val
│   │           ├── masks
│   │           │   ├──train
│   │           │   └──val
│   │           ├── polygons
│   │           │   ├──sem_seg_train.json
│   │           │   └──sem_seg_val.json
│   │           └── rles
│   │               ├──sem_seg_train.json
│   │               └──sem_seg_val.json
│   ├── nyu
│       ├── images
│       │   ├── train
│       │   ├── test
│       ├── annotations
│           ├── train
│           ├── test
```

## Training

To train the PIDNet model, use the training script with the desired configuration file:

```bash
python tools/train.py configs/pidnet/choose_a_config.py
```

Make sure to adjust the configuration file to match your dataset and training preferences.

If you encounter this error message

```bash
RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

And this is the fix: https://github.com/open-mmlab/mmsegmentation/issues/3724#issuecomment-2202124709

If you see any other error, don't hesitate to open an issue request. More support on https://github.com/open-mmlab/mmsegmentation/issues

## Real-Time Inference

To perform real-time inference using the `real_time_inference.py` script for video, image directories, or live camera feed, follow these steps:

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

### For Live Camera Feed Inference

To perform real-time segmentation using a live camera feed (e.g., webcam), use the following command:

```bash
python real_time_inference.py --camera 0 --config path/to/config.py --checkpoint path/to/checkpoint.pth --device cuda:0 --show
```

### Arguments

- `--video`: Path to the video file for inference.
- `--images`: Path to the directory containing images for inference.
- `--camera`: Camera source index (e.g., 0 for the default webcam).
- `--config`: Path to the model configuration file.
- `--checkpoint`: Path to the model checkpoint file.
- `--device`: Device to be used for inference (`cpu` or `cuda:0`).
- `--out`: Path to the output directory for images or video file path for saving results.
- `--show`: If specified, display the video or images during processing.
- `--wait-time`: Interval of show in seconds, default is 0.001 seconds.

This script will load the trained model, perform segmentation on the input video, images, or live camera feed, and display or save the results based on the provided arguments.

## Acknowledgements

- [PIDNet](https://github.com/XuJiacong/PIDNet)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
