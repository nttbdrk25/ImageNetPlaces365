# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 17:13:19 2023

@author: nguyen
"""
#model_name_dataset is format of "<modelname>_<datasetname>"
def get_model(model_name_dataset=None,num_class=None):
    str_tmp = model_name_dataset.split('_')
    model_name = str_tmp[0]
    datasetname = str_tmp[1]    
    if model_name == 'MAFNet':
        from MODELS import mobilenet_MAFC as MAFNet
        model = MAFNet.build_mobilenet_v1(num_class, width_multiplier=1, cifar=False,pool_types=['avg', 'std'])   
    if model_name == 'resnet18':
        from MODELS import model_resnet as res
        model = res.ResidualNet('ImageNet', 18, num_class, None)
    if model_name == 'mobilenetv1':
        from MODELS import mobilenet as MB
        model = MB.build_mobilenet_v1(num_class, width_multiplier=1.0)
    if model_name == 'mobilenetv2':
        from MODELS import mobilenet as MB
        model = MB.build_mobilenet_v2(num_class, width_multiplier=1.0)
    if model_name == 'mobilenetv3':
        from MODELS import mobilenet as MB
        model = MB.build_mobilenet_v3(num_class, "large", width_multiplier=1.0)
    if model_name == 'shufflenetv1':
        from MODELS import shufflenetV1 as SH        
        model = SH.ShuffleNetV1(n_class=num_class,model_size = '1.0x', group = 3)
    if model_name == 'shufflenetv2':
        from MODELS import shufflenetv2 as SH        
        model = SH.shufflenetv2(num_class=num_class)    
    if model_name == 'vgg16':
        from MODELS import vgg as V        
        model = V.vgg16(num_classes=num_class)    
    if model_name == 'GoogLeNet':
        from MODELS import GoogLeNet as G
        model = G.GoogLeNet(input_channel=3,n_classes=num_class)    
    if model_name == 'InceptionV4':
        from MODELS import InceptionV4 as Inc        
        model = Inc.InceptionV4(classes=num_class)
    return model