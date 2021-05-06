import pickle
import torch
import torch.nn as nn
from scipy import io
import numpy as np
import h5py
# fake_img = torch.randn(1,3,224,224)
# from torchvision.models import resnet101
# image_encoder = resnet101(pretrained=True)
# print(image_encoder.training)
# result1 = image_encoder(fake_img)
# image_encoder.eval()
# result2 = image_encoder(fake_img)
# diff = (result1 - result2).abs()
# print(diff.mean(), diff.std())
# image_encoder_until_avg_pool = torch.nn.Sequential(*(list(image_encoder.children())[:-2]))
# image_encoder_fc = torch.nn.Sequential(*(list(image_encoder.children())[-1:]))
# result2 = image_encoder_until_avg_pool(fake_img)
# result2 = nn.AdaptiveAvgPool2d((1,1))(result2)
# result2 = torch.flatten(result2, 1)
# result2 = image_encoder_fc(result2)
# diff = (result1 - result2).abs()
# print(diff.mean(), diff.std())
# # for name, param in image_encoder.named_parameters():
# #     print(name)
hf = h5py.File(f'../xlsa17/data/AWA2/res101_77_train_eval.h5', 'r')
res101 = io.loadmat('../xlsa17/data/AWA2/res101.mat')
att_splits = io.loadmat('../xlsa17/data/AWA2/att_splits.mat')
print(hf['features'].shape)
avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
for i in range(5):
    spatial = avgpool(torch.Tensor(hf['features'][...,i]).unsqueeze(0)).squeeze(2).squeeze(2)
    original = torch.Tensor(res101['features'][att_splits['train_loc'][0,i]])
    print(spatial.mean(), spatial.std(), spatial.max(), spatial.min())
    spatial = original
    print(spatial.mean(), spatial.std(), spatial.max(), spatial.min())
    # print((spatial-original).norm())