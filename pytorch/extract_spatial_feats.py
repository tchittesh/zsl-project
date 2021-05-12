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
resmatidx_to_imgname = {i: path[0].split('/')[-1] for i, path in enumerate(np.squeeze(res_mat['image_files']))}
loc_dict = {'train': 'train_loc', 'val': 'val_loc', 'test': 'test_unseen_loc'}
split_to_imgname_to_splitindex = {}
count = 0
for split, loc in loc_dict.items():
    split_to_imgname_to_splitindex[split] = {}
    count += len(att_splits[loc])
    for i, resmatidx in enumerate(np.squeeze(att_splits[loc]-1)):
        imgname = resmatidx_to_imgname[resmatidx]
        split_to_imgname_to_splitindex[split][imgname] = i
for split, imgname_to_splitindex in split_to_imgname_to_splitindex.items():
    assert set(range(len(imgname_to_splitindex))) == set(imgname_to_splitindex.values())

# Open h5 files for each split
#split_to_h5file = {}
#string_dtype = h5py.special_dtype(vlen=str)
#for split in loc_dict:
#    hf = h5py.File(f'../xlsa17/data/AWA2/res101_77_{split}.h5', 'w')
#    num_images = len(split_to_imgname_to_splitindex[split])
#    hf.create_dataset('features', (num_images, 2048, 7, 7), 'f')
#    #hf.create_dataset('features', (num_images, 2048), 'f')
#    hf.create_dataset('imgnames', (num_images,), string_dtype)
#    split_to_h5file[split] = hf

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
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

# Make ResNet model
image_encoder = get_original_caffe_resnet(spatial=True)
image_encoder.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_encoder.to(device)
alt_image_encoder = get_original_caffe_resnet(spatial=False)
alt_image_encoder.eval()
alt_image_encoder.to(device)

# Run ResNet and store results in h5 files
for idx, (imgs, labels, paths) in enumerate(tqdm(dataloader)):
    imgs = imgs.to(device)
    imgs = imgs * 255 - torch.Tensor(VGG_mean['image_mean']).permute(2,0,1).unsqueeze(0).cuda()
    imgs = imgs[:,(2,1,0),:,:]
    feats = image_encoder(imgs)
    alt_feats = alt_image_encoder(imgs).squeeze(2).squeeze(2)
    diff = (feats.mean(2).mean(2) - alt_feats).abs()
    print(diff.mean(), diff.std(), diff.min(), diff.max())
    for feat, path, label in zip(feats, paths, labels):
        imgname = path.split("/")[-1]
        for split, imgname_to_splitindex in split_to_imgname_to_splitindex.items():
            if imgname in imgname_to_splitindex:
                splitindex = imgname_to_splitindex[imgname]
                h5file = split_to_h5file[split]
                h5file['features'][splitindex,:,:,:] = feat.cpu().detach().numpy()
                h5file['imgnames'][splitindex] = imgname
                count -= 1
assert count == 0

# Close files
for h5file in split_to_h5file.values():
    h5file.close()
