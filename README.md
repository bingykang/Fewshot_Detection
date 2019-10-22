# Few-shot Object Detection via Feature Reweighting

Implementation for the paper:

[Few-shot Object Detection via Feature Reweighting](https://arxiv.org/abs/1812.01866), ICCV 2019

[Bingyi Kang](https://scholar.google.com.sg/citations?user=NmHgX-wAAAAJ)\*, [Zhuang Liu](https://liuzhuang13.github.io)\*, [Xin Wang](https://people.eecs.berkeley.edu/~xinw/), [Fisher Yu](https://www.yf.io), [Jiashi Feng](https://sites.google.com/site/jshfeng/home) and [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/) (\* equal contribution)

Our code is based on  [https://github.com/marvis/pytorch-yolo2](https://github.com/marvis/pytorch-yolo2) and developed with  Python 2.7 & PyTorch 0.3.1.





## Detection Examples (3-shot)

<div align=center>
<img src="https://user-images.githubusercontent.com/8370623/67256301-27d48e00-f43b-11e9-8348-b2b3fd1f5e99.png" width="740">
</div>

<div align=center>
Sample novel class detection results with 3-shot training bounding boxes, on PASCAL VOC.
</div> 

## Model
<div align=center>
<img src="https://user-images.githubusercontent.com/8370623/67256408-ad583e00-f43b-11e9-806e-47d79acecaed.png" width="740">
</div>

The architecture of our proposed few-shot detection model. It consists of a meta feature extractor and a reweighting module. The feature extractor follows the one-stage detector architecture and directly regresses the objectness score (o), bounding box location (x, y, h, w) and classification score (c). The reweighting module is trained to map support samples of N classes to N reweighting vectors, each responsible for modulating the meta features to detect the objects from the corresponding class. A softmax based classification score normalization is imposed on the final output.


## Abstract
Conventional training of a deep CNN based object detector demands a large number of bounding box annotations, which may be unavailable for rare categories. In this work we develop a few-shot object detector that can learn to detect novel objects from only a few annotated examples. Our proposed model leverages fully labeled base classes and quickly adapts to novel classes, using a meta feature learner and a reweighting module within a one-stage detection architecture. The feature learner extracts meta features that are generalizable to detect novel object classes, using training data from base classes with sufficient samples. The reweighting module transforms a few support examples from the novel classes to a global vector that indicates the importance or relevance of meta features for detecting the corresponding objects. These two modules, together with a detection prediction module, are trained end-to-end based on an episodic few-shot learning scheme and a carefully designed loss function. Through extensive experiments we demonstrate that our model outperforms well-established baselines by a large margin for few-shot object detection, on multiple datasets and settings. We also present analysis on various aspects of our proposed model, aiming to provide some inspiration for future few-shot detection works.





## Training our model on VOC

- ``` $PROJ_ROOT : project root ```
- ``` $DATA_ROOT : dataset root ```

### Prepare dataset
+ Get The Pascal VOC Data
```
cd $DATA_ROOT
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
```

+ Generate Labels for VOC
```
wget http://pjreddie.com/media/files/voc_label.py
python voc_label.py
cat 2007_train.txt 2007_val.txt 2012_*.txt > voc_train.txt
```

+ Generate per-class Labels for VOC (used for meta inpput)
```
cp $PROJ_ROOT/scripts/voc_label_1c.py $DATA_ROOT
cd $DATA_ROOT
python voc_label_1c.py
```

+ Generate few-shot image list
To use our few-shot datasets
```
cd $PROJ_ROOT
python scripts/convert_fewlist.py 
```

You may want to generate new few-shot datasets
Change the ''DROOT'' varibale in scripts/gen_fewlist.py to $DATA_ROOT
```
python scripts/gen_fewlist.py # might be different with ours
```

### Base Training
+ Modify Cfg for Pascal Data
Change the data/metayolo.data file 
```
metayolo=1
metain_type=2
data=voc
neg = 1
rand = 0
novel = data/voc_novels.txt             // file contains novel splits
novelid = 0                             // which split to use
scale = 1
meta = data/voc_traindict_full.txt
train = $DATA_ROOT/voc_train.txt
valid = $DATA_ROOT/2007_test.txt
backup = backup/metayolo
gpus=1,2,3,4
```

+ Download Pretrained Convolutional Weights
```
wget http://pjreddie.com/media/files/darknet19_448.conv.23
```

+ Train The Model
```
python train_meta.py cfg/metayolo.data cfg/darknet_dynamic.cfg cfg/reweighting_net.cfg darknet19_448.conv.23
```

+ Evaluate the Model
```
python valid_ensemble.py cfg/metayolo.data cfg/darknet_dynamic.cfg cfg/reweighting_net.cfg path/toweightfile
python scripts/voc_eval.py results/path/to/comp4_det_test_
```

### Few-shot Tuning
+ Modify Cfg for Pascal Data
Change the data/metatune.data file 
```
metayolo=1
metain_type=2
data=voc
tuning = 1
neg = 0
rand = 0
novel = data/voc_novels.txt                 
novelid = 0
max_epoch = 2000
repeat = 200
dynamic = 0
scale=1
train = $DATA_ROOT/voc_train.txt
meta = data/voc_traindict_bbox_5shot.txt
valid = $DATA_ROOT/2007_test.txt
backup = backup/metatune
gpus  = 1,2,3,4
```


+ Train The Model
```
python train_meta.py cfg/metatune.data cfg/darknet_dynamic.cfg cfg/reweighting_net.cfg path/to/base/weightfile
```

+ Evaluate the Model
```
python valid_ensemble.py cfg/metatune.data cfg/darknet_dynamic.cfg cfg/reweighting_net.cfg path/to/tuned/weightfile
python scripts/voc_eval.py results/path/to/comp4_det_test_
```

## Citation
```
@inproceedings{kang2019few,
  title={Few-shot Object Detection via Feature Reweighting},
  author={Kang, Bingyi and Liu, Zhuang and Wang, Xin and Yu, Fisher and Feng, Jiashi and Darrell, Trevor},
  booktitle={ICCV},
  year={2019}
}
```
