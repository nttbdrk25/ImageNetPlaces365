import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from .common import get_activation

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False, activation='relu'):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = get_activation(activation) if relu else None
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=None,activation='relu',excite_activation="sigmoid"):    
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.activation1 = get_activation(activation)
        self.activation2 = get_activation(excite_activation)
        if gate_channels // reduction_ratio == 0: #fixed for mobileNetV2
            reduction_ratio = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(len(pool_types)*gate_channels, gate_channels // reduction_ratio),            
            self.activation1,
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types        
    def forward(self, x):
        channel_att_sum = None
        squeeze_all=None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                squeeze = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            elif pool_type=='max':
                squeeze = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))                
            elif pool_type=='std':                
                stdf = torch.std(x,(2,3),unbiased=True)#compute standard deviation
                squeeze = stdf.reshape(stdf.size()[0],stdf.size()[1],1,1)#resize to be (,1,1) the same as out put of AdaptiveAvgPool2d , i.e., self.squeeze(residual)                
            if squeeze_all is None:
                squeeze_all = squeeze
            else:
                squeeze_all = torch.cat((squeeze_all,squeeze),1)                
        channel_att_sum = self.mlp(squeeze_all)    
        scale = self.activation2(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class ChannelPool(nn.Module):
    def __init__(self, pool_types=None):    
        super(ChannelPool, self).__init__()
        self.pool_types = pool_types
    def forward(self, x):
        spa_all = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                spa1 = torch.mean(x,1).unsqueeze(1)                
            if pool_type=='max':
                spa1 = torch.max(x,1)[0].unsqueeze(1)
            if pool_type=='std':
                spa1 = torch.std(x,1).unsqueeze(1)
            if spa_all is None:
                spa_all = spa1
            else:
                spa_all = torch.cat((spa_all, spa1), dim=1)
        return spa_all
class SpatialGate(nn.Module):
    def __init__(self,pool_types=None):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool(pool_types)
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class MAFC(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=None, no_spatial=False,activation='relu',excite_activation="sigmoid"):
        super(MAFC, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types,activation=activation,excite_activation=excite_activation)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate(pool_types)
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
