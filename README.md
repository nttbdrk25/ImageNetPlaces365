# ReINs and RePLs: Challenging small datasets rescaled from ImageNet and Places365
**Abstract:**

* Efficiently rescaling a large dataset by adapting statistical
computation to the validation outcomes of a pre-trained network.
* A unified collection of the sensitive images and those in
their confused classes would form a challenging tiny set.
* An application to rescaling two large datasets: ImageNet
and Places365 to obtain their rescaled subsets.
* Experimental results for image classification have validated
the raised challenge of the rescaled subsets. Verifying on
these subsets helps researchers to save time and the computational
cost for the early network drafts.
* It can be conducive to a good rate of a network draft on the
large datasets if it is good on the rescaled subsets.
# Rescaling explanations
* **Names of rescaled classes of ImageNet:** 30 classes: ReIN<sup>30</sup> (![ReIN30.txt](Result_Rescaled_ImageNet/ReIN30.txt)), 50 classes: ReIN<sup>50</sup> (![ReIN50.txt](Result_Rescaled_ImageNet/ReIN50.txt)), 100 classes: ReIN<sup>100</sup> (![ReIN100.txt](Result_Rescaled_ImageNet/ReIN100.txt)), 150 classes: ReIN<sup>150</sup> (![ReIN150.txt](Result_Rescaled_ImageNet/ReIN150.txt)), and 200 classes: ReIN<sup>200</sup> (![ReIN200.txt](Result_Rescaled_ImageNet/ReIN200.txt)) in folder: *`Result_Rescaled_ImageNet`*
![List of rescaled subsets of ImageNet (pdf)](Result_Rescaled_ImageNet/Rescaled_Subsets_ImageNet.pdf)
* **Names of rescaled classes of Places365:** 30 classes: RePL<sup>30</sup> (![RePL30.txt](Result_Rescaled_Places365/RePL30.txt)) and 50 classes: RePL<sup>50</sup> (![RePL50.txt](Result_Rescaled_Places365/RePL50.txt)) in folder: *`Result_Rescaled_Places365`*
![List of rescaled subsets of Places365 (pdf)](Result_Rescaled_Places365/Rescaled_Subsets_Places365.pdf) 

* **Retrieve rescaled subsets from the large datasets ImageNet/Places365 by executing this command**
```
$ python RescaleDataset.py
```
Note: Default is for ImageNet. Change *`DataRescale = 'Places365'`* in file *`config_rescale.py`* for Places365

* **For training some CNN-based networks on the rescaled sub-datasets of ImageNet**
```
$ python Train_ImageNetRescaleSubsets.py
```
* **For training some CNN-based networks on the rescaled sub-datasets of Places365**
```
$ python Train_Places365RescaleSubsets.py
```
# Experimental results (initial), Top1-accuracy (%)
|Network|ReIN<sup>30</sup>|ReIN<sup>50</sup>|ReIN<sup>100</sup>|ReIN<sup>150</sup>|ReIN<sup>200</sup>|ImageNet|RePL<sup>30</sup>|RePL<sup>50</sup>|Places365
| ------------- | -------------: |-------------: | -------------: |-------------: | -------------: |-------------: | -------------: |-------------: | -------------: |
|MAFNet|48.40|50.24|54.02|56.05|57.26|73.13|52.57|52.50|55.15|
|MobileNetV1|46.53|50.08|52.48|55.84|56.47|70.60|51.53|50.28|53.50|
|MobileNetV2|47.47|49.08|53.74|54.37|55.37|72.00|52.67|52.02|52.19|
|MobileNetV3|49.87|51.68|54.06|55.56|55.72|71.50|52.33|51.68|53.53|
|ShuffleNetV2|48.87|49.52|51.38|52.05|53.57|69.36|50.97|51.12|50.80|
|ResNet18|47.67|50.08|53.38|55.01|56.36|70.40|51.30|52.28|54.43|
|GoogLeNet|44.40|48.92|49.90|51.36|53.20|68.30|50.80|50.62|53.63|
|InceptionV4|48.48|49.24|49.67|50.77|51.81|80.00|49.34|50.00|51.92|
|VGG16|51.13|51.48|52.72|54.30|54.85|75.20|52.50|52.24|55.24|
# Related citations
If you use any materials, please cite the following relevant works.
```
@article{prlNguyen23,
  author       = {Thanh Tuan Nguyen and Thanh Phuong Nguyen},
  title        = {Rescaling Large Datasets Based on Validation Outcomes of a Pre-trained Network},
  journal      = {Pattern Recognition Letters},
  note         = {(accepted in July 2024)}
}
```

```
@inproceedings{cvprDengDSLL009,
  author       = {Jia Deng and Wei Dong and Richard Socher and Li{-}Jia Li and Kai Li and Li Fei{-}Fei},
  title        = {ImageNet: {A} large-scale hierarchical image database},
  booktitle    = {CVPR},
  pages        = {248--255},  
  year         = {2009}
}
```

```
@article{pamiZhouLKO018,
  author    = {Bolei Zhou and {\`{A}}gata Lapedriza and Aditya Khosla and Aude Oliva and Antonio Torralba},
  title     = {Places: {A} 10 Million Image Database for Scene Recognition},
  journal   = {{IEEE} Trans. Pattern Anal. Mach. Intell.},
  volume    = {40},
  number    = {6},
  pages     = {1452--1464},
  year      = {2018}
}
```

```
@article{paaNguyen24,
  author       = {Thanh Tuan Nguyen and Thanh Phuong Nguyen and Vincent Nguyen},
  title        = {Accumulating global channel-wise patterns via deformed-bottleneck recalibration},
  journal      = {Pattern Analysis and Applications},
  note         = {(accepted in 2025)}
}
```

