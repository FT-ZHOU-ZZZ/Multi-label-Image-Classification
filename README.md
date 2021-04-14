# Dataset Formatter and Experimential Results for Multi-label Image Classification Datasets
This repository provides the <b>Downloader</b>, <b>Formatter</b> and  <b>Experimental Results</b> for <b>Multi-label Image Classification</b>, which is implemented with PyTorch.<br />
Meanwhile, we collect and present the multi-label image classification results published in recent years.<br />
If you have any questions about this repository, please do not hesitate to contact me by emails (<b><u>zft@cqu.edu.cn</u></b>).

## Popular Datasets about Multi-label Image Classification
### [Pascal VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/).
![avatar](Pascal-VOC2007/example.png)
Pascal VOC 2007 dataset is a benchmark in the field of computer vision, which is widely used in image classification, 
object detection, target segmentation and other tasks, which contains four categories: vehicle, household, animal, 
person, and can be subdivided into **20 classes**. For multi-label image classification, VOC 2007 dataset is divided into 
**training (2,501)**, **validation (2,510)** and **testing (4,952)** sets. Following the previous work, we usually use the train and 
validate set to train our model, and evaluate the classification performance on the test set.<br />
See the [VOC2007 folder]() for more details!

### [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).
![avatar](Pascal-VOC2012/example.png)
 The classes of VOC 2012 are same as VOC 2007, but the 2012 dataset contains images from 2008-2011 and there is no 
 intersection with 2007. VOC 2012 dataset is divided into **training (5,717)**, **validation (5,823)** and **testing (10,991)** set. 
 However, no ground truth labels are provided in the test set. Therefore, all the approaches have to be evaluated by 
 submitting the testing results to the PASCAL VOC Evaluation Server. We train our model on train set, and fine-tune on 
 validate set. Then, the result of test set is submitted to **[Evaluation Server](http://host.robots.ox.ac.uk:8080/)** for evaluation.<br />
See the [VOC2012 folder]() for more details!

### [Microsoft COCO](https://cocodataset.org/).
![avatar](MS-COCO/example.png)
Microsoft COCO is large scale images with Common Objects in Context (COCO), which is one of the most popular image 
datasets out there, with applications like object detection, segmentation, and image caption. The images in the dataset 
are everyday objects captured from everyday scenes. COCO provides multi-object labeling, segmentation mask annotations, 
image captioning, key-point detection and panoptic segmentation annotations with a total of 81 categories, making it a very 
versatile and multi-purpose dataset. More concretely, the dataset contains **122,218 images** and covers **80 common 
categories** (class 81 is background), which is further divided into a **training set of 82,081 images** and a **validation set 
of 40,137 images**. Since the ground truth annotations of test set are unavailable, we usually train our model on the training 
datasets and evaluate on the validation set. <br />
See the [MS-COCO folder]() for more details!


## Tips
More downloader, formatter and experimental results about multi-label image classification will come soon.<br />
If the repository helps you, please star it, thanks!