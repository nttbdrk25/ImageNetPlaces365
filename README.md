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
these subsets helps researchers to save the computational
cost for the early network drafts.
* It can be conducive to a good rate of a network draft on the
large datasets if it is good on the rescaled subsets.
# Rescaling explanations
```
$ python RescaleDataset.py
```

```
$ python Train_ImageNetRescaleSubsets.py #for training some CNN-based networks on the rescaled sub-datasets of ImageNet
or
$ python Train_Places365RescaleSubsets.py #for training some CNN-based networks on the rescaled sub-datasets of Places365
```
# Related citations
If you use any material, please cite relevant works as follows.
```
@article{apinNguyen23,
  author       = {Thanh Tuan Nguyen and
                  Thanh Phuong Nguyen and
                  Vincent Nguyen},
  title        = {A deformed bottleneck recalibration for accumulating global channel-wise features},
  journal      = {Applied Intelligence},
  note         = {(submitted in May 2023)}
}
@inproceedings{cvprDengDSLL009,
  author       = {Jia Deng and Wei Dong and Richard Socher and Li{-}Jia Li and Kai Li and Li Fei{-}Fei},
  title        = {ImageNet: {A} large-scale hierarchical image database},
  booktitle    = {CVPR},
  pages        = {248--255},  
  year         = {2009}
}
```
