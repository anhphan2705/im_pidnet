```
mmsegmentation
├── mmseg
├── tools
├── configs
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── VOCdevkit
│   │   ├── VOC2012
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClass
│   │   │   ├── ImageSets
│   │   │   │   ├── Segmentation
│   │   ├── VOC2010
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClassContext
│   │   │   ├── ImageSets
│   │   │   │   ├── SegmentationContext
│   │   │   │   │   ├── train.txt
│   │   │   │   │   ├── val.txt
│   │   │   ├── trainval_merged.json
│   │   ├── VOCaug
│   │   │   ├── dataset
│   │   │   │   ├── cls
│   ├── ade
│   │   ├── ADEChallengeData2016
│   │   │   ├── annotations
│   │   │   │   ├── training
│   │   │   │   ├── validation
│   │   │   ├── images
│   │   │   │   ├── training
│   │   │   │   ├── validation
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
│   ├── CHASE_DB1
│   │   ├── images
│   │   │   ├── training
│   │   │   ├── validation
│   │   ├── annotations
│   │   │   ├── training
│   │   │   ├── validation
│   ├── DRIVE
│   │   ├── images
│   │   │   ├── training
│   │   │   ├── validation
│   │   ├── annotations
│   │   │   ├── training
│   │   │   ├── validation
│   ├── HRF
│   │   ├── images
│   │   │   ├── training
│   │   │   ├── validation
│   │   ├── annotations
│   │   │   ├── training
│   │   │   ├── validation
│   ├── STARE
│   │   ├── images
│   │   │   ├── training
│   │   │   ├── validation
│   │   ├── annotations
│   │   │   ├── training
│   │   │   ├── validation
|   ├── dark_zurich
|   │   ├── gps
|   │   │   ├── val
|   │   │   └── val_ref
|   │   ├── gt
|   │   │   └── val
|   │   ├── LICENSE.txt
|   │   ├── lists_file_names
|   │   │   ├── val_filenames.txt
|   │   │   └── val_ref_filenames.txt
|   │   ├── README.md
|   │   └── rgb_anon
|   │   |   ├── val
|   │   |   └── val_ref
|   ├── NighttimeDrivingTest
|   |   ├── gtCoarse_daytime_trainvaltest
|   |   │   └── test
|   |   │       └── night
|   |   └── leftImg8bit
|   |   |   └── test
|   |   |       └── night
│   ├── loveDA
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   ├── potsdam
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   ├── vaihingen
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   ├── iSAID
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   ├── synapse
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   ├── REFUGE
│   │   ├── images
│   │   │   ├── training
│   │   │   ├── validation
│   │   │   ├── test
│   │   ├── annotations
│   │   │   ├── training
│   │   │   ├── validation
│   │   │   ├── test
│   ├── mapillary
│   │   ├── training
│   │   │   ├── images
│   │   │   ├── v1.2
|   │   │   │   ├── instances
|   │   │   │   ├── labels
|   │   │   │   └── panoptic
│   │   │   ├── v2.0
|   │   │   │   ├── instances
|   │   │   │   ├── labels
|   │   │   │   ├── panoptic
|   │   │   │   └── polygons
│   │   ├── validation
│   │   │   ├── images
|   │   │   ├── v1.2
|   │   │   │   ├── instances
|   │   │   │   ├── labels
|   │   │   │   └── panoptic
│   │   │   ├── v2.0
|   │   │   │   ├── instances
|   │   │   │   ├── labels
|   │   │   │   ├── panoptic
|   │   │   │   └── polygons
│   ├── bdd100k
│   │   ├── images
│   │   │   └── 10k
|   │   │   │   ├── test
|   │   │   │   ├── train
|   │   │   │   └── val
│   │   └── labels
│   │   │   └── sem_seg
|   │   │   │   ├── colormaps
|   │   │   │   │   ├──train
|   │   │   │   │   └──val
|   │   │   │   ├── masks
|   │   │   │   │   ├──train
|   │   │   │   │   └──val
|   │   │   │   ├── polygons
|   │   │   │   │   ├──sem_seg_train.json
|   │   │   │   │   └──sem_seg_val.json
|   │   │   │   └── rles
|   │   │   │   │   ├──sem_seg_train.json
|   │   │   │   │   └──sem_seg_val.json
│   ├── nyu
│   │   ├── images
│   │   │   ├── train
│   │   │   ├── test
│   │   ├── annotations
│   │   │   ├── train
│   │   │   ├── test
│   ├── HSIDrive20
│   │   ├── images
│   │   │   ├── train
│   │   │   ├── validation
│   │   │   ├── test
│   │   ├── annotations
│   │   │   ├── train
│   │   │   ├── validation
│   │   │   ├── test
```