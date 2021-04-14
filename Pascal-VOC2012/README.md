# Pascal VOC 2012

## Introduction
Pascal VOC 2012 dataset is a benchmark in the field of computer vision, which is widely used in image classification,
object detection, target segmentation and other tasks, which contains four categories: Vehicle, Indoor, Animal,
Person, and can be subdivided into **20 classes**.

* Person: person
* Animal: bird, cat, cow, dog, horse, sheep
* Vehicle: aeroplane, bicycle, boat, bus, car, motorbike, train
* Indoor: bottle, chair, dining table, potted plant, sofa, tv/monitor

![avatar](example.png)

The classes of VOC 2012 are same as VOC 2007, but the 2012 dataset contains images from 2008-2011 and there is no 
 intersection with 2007. VOC 2012 dataset is divided into **training (5,717)**, **validating (5,823)** and **testing (10,991)** set. 
 However, no ground truth labels are provided in the test set. Therefore, all the approaches have to be evaluated by 
 submitting the testing results to the PASCAL VOC Evaluation Server. We usually train our model on training set, and 
 fine-tune on validate set. Then, the result of testing set is submitted to 
 **[Evaluation Server](http://host.robots.ox.ac.uk:8080/)** for evaluation.<br />
You can see more detailed description of this dataset on its **[website](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)**. <br />
If you have any questions about this repository, please do not hesitate to contact me by emails (<b><u>zft@cqu.edu.cn</u></b>).

## Run
```sh
python3 main_voc2012.py
```

## Output
```sh
[dataset] read data\files\VOC2012\classification_train.csv
[dataset] VOC 2012 classification set=train number of classes=20  number of images=5717
[dataset] read data\files\VOC2012\classification_val.csv
[dataset] VOC 2012 classification set=val number of classes=20  number of images=5823
[dataset] read data\files\VOC2012\classification_test.csv
[dataset] VOC 2012 classification set=test number of classes=20  number of images=10991
```

## Tips
Due to the networks reasons or other unexpected circumstances, it may not be able to download the files automatically.
Please download the following files manually and put them in this folder (`$DATA_PATH$`), then `run main_voc2012.py`.

* `VOCtrainval_11-May-2012.tar`: training/validation data ([download](https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar))
* `VOC2012test.tar`: testing data without annotation ([download](https://pjreddie.com/media/files/VOC2012test.tar))
* `VOCdevkit_18-May-2011.tar`: development kit code and documentation ([download](http://pjreddie.com/media/files/VOCdevkit_18-May-2011.tar))

## Note
No ground truth labels are provided in the testing set. We must format the results of each category in testing 
set as following
```sh
comp1_cls_test_aeroplane.txt
comp1_cls_test_bicycle.txt
comp1_cls_test_bird.txt
.
.
.
comp1_cls_test_tv.txt
```
And, then submit the results to [Evaluation Server](http://host.robots.ox.ac.uk:8080/) for evaluation.<br />
Please, see [Evaluation Server](http://host.robots.ox.ac.uk:8080/) for more details.

## Experimental Results
These are some works in recent years (rank by time, and only include the papers published in the top journals or top conferences).
      
| No. | Conference |    Method    | aero  |  bike |  bird |  boat | bottle|  bus  |  car  |  cat  | chair |  cow  | table |  dog  | horse | motor | person| plant | sheep |  sofa | train |   tv  | mAP |
|:---:|:----------:|:------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:---:|
|  1  |ICLR-2014   |VGG16+SVM     | 99.0  | 88.8  | 95.9  | 93.8  | 73.1  | 92.1  | 85.1  | 97.8  | 79.5  | 91.1  | 83.3  | 97.2  | 96.3  | 94.5  | 96.9  | 63.1  | 93.4  | 75.0  | 97.1  | 87.1  | 89.0|
|  1  |ICLR-2014   |VGG19+SVM     | 99.1  | 88.7  | 95.7  | 93.9  | 73.1  | 92.1  | 84.8  | 97.7  | 79.1  | 90.7  | 83.2  | 97.3  | 96.2  | 94.3  | 96.9  | 63.4  | 93.2  | 74.6  | 97.3  | 87.9  | 89.0|
|  2  |TPAMI-2015  |HCP           | 99.1  | 92.8  | 97.4  | 94.4  | 79.9  | 93.6  | 89.8  | 98.2  | 78.2  | 94.9  | 79.8  | 97.8  | 97.0  | 93.8  | 96.4  | 74.3  | 94.7  | 71.9  | 96.7  | 88.6  | 90.5|
|  3  |CVPR-2016   |FeV+LV        | 98.4  | 92.8  | 93.4  | 90.7  | 74.9  | 93.2  | 90.2  | 96.1  | 78.2  | 89.8  | 80.6  | 95.7  | 96.1  | 95.3  | 97.5  | 73.1  | 91.2  | 75.4  | 97.0  | 88.2  | 89.4|
|  4  |TIP-2016	   |RCP           | 99.3  | 92.2  | 97.5  | 94.9  | 82.3  | 94.1  | 92.4  | 98.5  | 83.8  | 93.5  | 83.1  | 98.1  | 97.3  | 96.0  | 98.8  | 77.7  | 95.1  | 79.4  | 97.7  | 92.4  | 92.2|
|  5  |AAAI-2018   |RMIC          | 98.0  | 85.5  | 92.6  | 88.7  | 64.0  | 86.8  | 82.0  | 94.9  | 72.7  | 83.1  | 73.4  | 95.2  | 91.7  | 90.8  | 95.5  | 58.3  | 87.6  | 70.6  | 93.8  | 83.0  | 84.4|
|  6  |TMM-2018    |RLSD          | 96.4  | 92.7  | 93.8  | 94.1  | 71.2  | 92.5  | 94.2  | 95.7  | 74.3  | 90.0  | 74.2  | 95.4  | 96.2  | 92.1  | 97.9  | 66.9  | 93.5  | 73.7  | 97.5  | 87.6  | 88.5|
|  7  |PR-2019	   |DELTA         |   -   |   -	  |   -	  |   -	  |   -	  |   -	  |   -	  |   -	  |   -	  |   -	  |   -	  |   -	  |   -	  |   -	  |   -	  |   -	  |   -	  |   -   |   -	  |   -	  | 90.3|
|  8  |ICCV-2019   |SSGRL         | 99.5  | 95.1  | 97.4  | 96.4  | 85.8  | 94.5  | 93.7  | 98.9  | 86.7  | 96.3  | 84.6  | 98.9  | 98.6  | 96.2  | 98.7  | 82.2  | 98.2  | 84.2  | 98.1  | 93.5  | 93.9|
|  8  |ICCV-2019   |SSGRL-Pre     | 99.7  | 96.1  | 97.7  | 96.5  | 86.9  | 95.8  | 95.0  | 98.9  | 88.3  | 97.6  | 87.4  | 99.1  | 99.2  | 97.3  | 99.0  | 84.8  | 98.3  | 85.8  | 99.2  | 94.1  | 94.8|
|  9  |ECCV-2020   |ADD-GCN       | 99.8  | 97.1  | 98.6  | 96.8  | 89.4  | 97.1  | 96.5  | 99.3  | 89.0  | 97.7  | 87.5  | 99.2  | 99.1  | 97.7  | 99.1  | 86.3  | 98.8  | 87.0  | 99.3  | 95.4  | 95.5|
|  10 |AAAI-2021   |DSDL          | 99.4  | 95.3  | 97.6  | 95.7  | 83.5  | 94.8  | 93.9  | 98.5  | 85.7  | 94.5  | 83.8  | 98.4  | 97.7  | 95.9  | 98.5  | 80.6  | 95.7  | 82.3  | 98.2  | 93.2  | 93.2|


## References
1. Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition[J]. arXiv preprint arXiv:1409.1556, 2014.<br />
2. Wei Y, Xia W, Lin M, et al. HCP: A flexible CNN framework for multi-label image classification[J]. IEEE transactions on pattern analysis and machine intelligence, 2015, 38(9): 1901-1907.<br />
3. Yang H, Tianyi Zhou J, Zhang Y, et al. Exploit bounding box annotations for multi-label object recognition[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 280-288.<br />
4. Wang M, Luo C, Hong R, et al. Beyond object proposals: Random crop pooling for multi-label image recognition[J]. IEEE Transactions on Image Processing, 2016, 25(12): 5678-5688.<br />
5. He S, Xu C, Guo T, et al. Reinforced multi-label image classification by exploring curriculum[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2018, 32(1).<br />
6. Zhang J, Wu Q, Shen C, et al. Multilabel image classification with regional latent semantic dependencies[J]. IEEE Transactions on Multimedia, 2018, 20(10): 2801-2813.<br />
7. Yu W J, Chen Z D, Luo X, et al. DELTA: A deep dual-stream network for multi-label image classification[J]. Pattern Recognition, 2019, 91: 322-331.<br />
8. Chen T, Xu M, Hui X, et al. Learning semantic-specific graph representation for multi-label image recognition[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019: 522-531.<br />
9. Ye J, He J, Peng X, et al. Attention-Driven Dynamic Graph Convolutional Network for Multi-Label Image Recognition[C]//European Conference on Computer Vision. Springer, Cham, 2020: 649-665.<br />
10. Zhou F, Huang S, Xing Y. Deep Semantic Dictionary Learning for Multi-label Image Classification[J]. arXiv preprint arXiv:2012.12509, 2020.<br />