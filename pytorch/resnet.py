import imp
import torch
from torchvision.models import resnet101

def get_original_caffe_resnet(spatial=False):
    if spatial:
        MainModel = imp.load_source('MainModel', "../models/pytorch_spatial_resnet.py")
    else:
        MainModel = imp.load_source('MainModel', "../models/pytorch_features_resnet.py")
    model = torch.load('../models/pytorch_resnet.pth')
    return model

def get_torchvision_resnet(spatial=False):
    model = resnet101(pretrained=True)
    if spatial:
        return torch.nn.Sequential(*(list(model.children())[:-2]))
    else:
        return torch.nn.Sequential(*(list(model.children())[:-1]))

def get_hub_resnet(spatial=False):
    model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet101', pretrained=True)
    if spatial:
        return torch.nn.Sequential(*(list(model.children())[:-2]))
    else:
        return torch.nn.Sequential(*(list(model.children())[:-1]))
