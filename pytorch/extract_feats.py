import os

import h5py
import numpy as np
import scipy.io as io
import torchvision
import torchvision.transforms as transforms
import torch
from tqdm import tqdm

from resnet import get_original_caffe_resnet, get_torchvision_resnet, get_hub_resnet

att_splits = io.loadmat('../xlsa17/data/AWA2/att_splits.mat')
res_mat = io.loadmat('../xlsa17/data/AWA2/res101.mat')
VGG_mean = io.loadmat('/mnt/data/VGG_mean.mat')

# Generate the mapping from split -> imgname -> index in split
imgname_to_resmatidx = {path[0].split('/')[-1]: i for i, path in enumerate(np.squeeze(res_mat['image_files']))}
res_mat_manual = np.zeros((len(imgname_to_resmatidx), 2048), dtype=np.float)
count = len(res_mat_manual)

# Make dataloader
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
#    transforms.Resize([224, 224]),
    transforms.ToTensor(),
])
def datasetfolder_getitem(self, index):
    path, target = self.samples[index]
    sample = self.loader(path)
    if self.transform is not None:
        sample = self.transform(sample)
    if self.target_transform is not None:
        target = self.target_transform(target)

    return sample, target, path
torchvision.datasets.DatasetFolder.__getitem__ = datasetfolder_getitem
dataset = torchvision.datasets.ImageFolder(
        root='/mnt/data/Animals_with_Attributes2/JPEGImages/',
        transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

# Make ResNet model
image_encoder = get_original_caffe_resnet(spatial=False)
image_encoder.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_encoder.to(device)

# Run ResNet and store results in h5 files
debug = False
for idx, (imgs, labels, paths) in enumerate(tqdm(dataloader)):
    #imgs = imgs[:,(2,1,0),:,:].to(device)
    imgs = imgs.to(device)
    imgs = imgs * 255 - torch.Tensor(VGG_mean['image_mean']).permute(2,0,1).unsqueeze(0).cuda()
    imgs = imgs[:,(2,1,0),:,:]
    feats = image_encoder(imgs).squeeze(2).squeeze(2)
    for feat, path, label in zip(feats, paths, labels):
        imgname = path.split("/")[-1]
        resmatidx = imgname_to_resmatidx[imgname]
        res_mat_manual[resmatidx,:] = feat.cpu().detach().numpy()
        if debug:
            print('----------')
            print(path)
            feat = res_mat_manual[resmatidx,:]
            print(feat.mean(), feat.std(), feat.min(), feat.max())
            feat = res_mat['features'][:,resmatidx]
            print(feat.mean(), feat.std(), feat.min(), feat.max())
            print(res_mat['image_files'][resmatidx])
            input()
        count -= 1

io.savemat("../xlsa17/data/AWA2/res101_manual.mat", {
    "features": res_mat_manual.transpose(1,0),
    "image_files": res_mat["image_files"], 
    "labels": res_mat["labels"]
})

assert count == 0
