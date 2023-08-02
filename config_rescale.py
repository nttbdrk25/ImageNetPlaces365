# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 13:45:47 2023

@author: nguyen
"""
DataRescale = 'ImageNet' #DataRescale has to be Places365 or ImageNet
if DataRescale == 'ImageNet':
    path_in_dataset = '../../../datasets/ImageNetForRescale/ImageNet/'#path to the large dataset needed to be rescaled
    path_file_rescale = './Result_Rescaled_ImageNet/' #containter .txt files of rescaling results
    path_out_rescale = './ImageNet_Rescaled_Subsets/'#destination for storing the rescaled subsets
if DataRescale == 'Places365':
    path_in_dataset = '../../../datasets/places365standard/places365_standard/'
    path_file_rescale = './Result_Rescaled_Places365/'
    path_out_rescale = './Places365_Rescaled_Subsets/'