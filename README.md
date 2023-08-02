# ImageNetPlaces365 (Rescaling large datasets)
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
* **Names of rescaled classes of ImageNet:** ReIN30, ReIN50, ReIN100, ReIN150, and ReIN200 in folder: *Result_Rescaled_ImageNet*
* **Names of rescaled classes of Places365:** RePL30 and RePL50 in folder: '*Result_Rescaled_Places365*'

* **Retrieve rescaled subsets from the large datasets ImageNet/Places365 by executing this command**
```
$ python RescaleDataset.py
```
Note: Default is for ImageNet. Change *DataRescale = 'Places365'* in file *config_rescale.py* for Places365

* **For training some CNN-based networks on the rescaled sub-datasets of ImageNet**
```
$ python Train_ImageNetRescaleSubsets.py
```
* **For training some CNN-based networks on the rescaled sub-datasets of Places365**
```
$ python Train_Places365RescaleSubsets.py
```
# Experimental results (initial)
|Network|ReIN30|ReIN50|ReIN100|ReIN150|ReIN20|ImageNet|RePL30 RePL50|Places365
| ------------- | ------------- |
# Related citations
If you use any material, please cite relevant works as follows.
```
@article{prlNguyen23,
  author       = {Thanh Tuan Nguyen and Thanh Phuong Nguyen},
  title        = {Rescaling Large Datasets Based on Validation Outcomes of a Pre-trained Network},
  journal      = {Pattern Recognition Letters},
  note         = {(submitted in Aug 2023)}
}
```

```
@article{apinNguyen23,
  author       = {Thanh Tuan Nguyen and Thanh Phuong Nguyen and Vincent Nguyen},
  title        = {A deformed bottleneck recalibration for accumulating global channel-wise features},
  journal      = {Applied Intelligence},
  note         = {(submitted in May 2023)}
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
