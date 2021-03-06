import numpy as np
from scipy import io, spatial
import torch
import h5py

from utils import normalizeFeaturesL2

AWA2_CLASSES = ['antelope', 'grizzly+bear', 'killer+whale', 'beaver', 'dalmatian', 'persian+cat', 'horse', 'german+shepherd', 'blue+whale', 'siamese+cat', 'skunk', 'mole', 'tiger', 'hippopotamus', 'leopard', 'moose', 'spider+monkey', 'humpback+whale', 'elephant', 'gorilla', 'ox', 'fox', 'sheep', 'seal', 'chimpanzee', 'hamster', 'squirrel', 'rhinoceros', 'rabbit', 'bat', 'giraffe', 'wolf', 'chihuahua', 'rat', 'weasel', 'otter', 'buffalo', 'zebra', 'giant+panda', 'deer', 'bobcat', 'pig', 'lion', 'mouse', 'polar+bear', 'collie', 'walrus', 'raccoon', 'cow', 'dolphin']

AWA2_ATTRIBUTES = ['black', 'white', 'blue', 'brown', 'gray', 'orange', 'red', 'yellow', 'patches', 'spots', 'stripes', 'furry', 'hairless', 'toughskin', 'big', 'small', 'bulbous', 'lean', 'flippers', 'hands', 'hooves', 'pads', 'paws', 'longleg', 'longneck', 'tail', 'chewteeth', 'meatteeth', 'buckteeth', 'strainteeth', 'horns', 'claws', 'tusks', 'smelly', 'flys', 'hops', 'swims', 'tunnels', 'walks', 'fast', 'slow', 'strong', 'weak', 'muscle', 'bipedal', 'quadrapedal', 'active', 'inactive', 'nocturnal', 'hibernate', 'agility', 'fish', 'meat', 'plankton', 'vegetation', 'insects', 'forager', 'grazer', 'hunter', 'scavenger', 'skimmer', 'stalker', 'newworld', 'oldworld', 'arctic', 'coastal', 'desert', 'bush', 'plains', 'forest', 'fields', 'jungle', 'mountains', 'ocean', 'ground', 'water', 'tree', 'cave', 'fierce', 'timid', 'smart', 'group', 'solitary', 'nestspot', 'domestic']

class ZSLDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_name, split, norm_type='none', norm_info=None):
        assert dataset_name in ('APY', 'AWA1', 'AWA2', 'CUB', 'SUN')
        loc_dict = {'train': 'train_loc', 'val': 'val_loc', 'test': 'test_unseen_loc'}
        assert split in loc_dict
        loc = loc_dict[split]
        
        # load data
        data_folder = '../xlsa17/data/'+dataset_name+'/'
        res101 = io.loadmat(data_folder+'res101.mat')
        att_splits=io.loadmat(data_folder+'att_splits.mat')

        # filter based on split
        self.img_features = torch.Tensor(res101['features'][:, np.squeeze(att_splits[loc]-1)]).permute(1,0) # shape [N,d]
        self.img_names = [i[0].split('/')[-1] for i in res101['image_files'][np.squeeze(att_splits[loc])-1,0]]
        self.labels = torch.LongTensor(np.squeeze(res101['labels'][np.squeeze(att_splits[loc]-1)])) # shape [N]    
        unique_labels = np.unique(self.labels)
        self.attributes = AWA2_ATTRIBUTES
        self.classes = np.array(AWA2_CLASSES)[unique_labels-1]
        i=0
        for label in unique_labels:
            self.labels[self.labels == label] = i
            i+=1
        self.class_attributes = torch.Tensor(att_splits['att'][:, unique_labels-1]) # shape [num_attributes, num_classes]

        self.length = len(self.labels)
        assert self.length == self.img_features.shape[0]

        assert norm_type in ('std', 'L2', 'None')
        if norm_type == 'std':
            if split == 'train':
                self.norm_info = {
                    'std': self.img_features.std(0),
                    'mean': self.img_features.mean(0),
                }
            else:
                assert norm_info is not None
                self.norm_info = norm_info
            std = self.norm_info['std'].unsqueeze(0)
            std[std == 0] = 1
            mean = self.norm_info['mean'].unsqueeze(0)
            self.img_features = (self.img_features - mean) / std
        elif norm_type == 'L2':
            self.img_features = normalizeFeaturesL2(self.img_features)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.img_features[idx,:]
        label = self.labels[idx]
        class_attributes = self.class_attributes[:,label]
        return {'img': img, 'label': label, 'class_attributes': class_attributes}

    def get_img_path(self, idx):
        img_name = self.img_names[idx]
        class_name = img_name.split('_')[0]
        return f"/mnt/data/Animals_with_Attributes2/JPEGImages/{class_name}/{img_name}"

    def get_num_attributes(self):
        return self.class_attributes.shape[0]

    def get_img_feature_size(self):
        return self.img_features.shape[1]


class ZSLSpatialDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_name, split, spatial=False, norm_type='none', norm_info=None):
        assert dataset_name in ('APY', 'AWA1', 'AWA2', 'CUB', 'SUN')
        loc_dict = {'train': 'train_loc', 'val': 'val_loc', 'test': 'test_unseen_loc'}
        assert split in loc_dict
        loc = loc_dict[split]
        
        # load data
        data_folder = '../xlsa17/data/'+dataset_name+'/'
        res101 = io.loadmat(data_folder+'res101.mat')
        att_splits=io.loadmat(data_folder+'att_splits.mat')

        # filter based on split
        h5_file = h5py.File(data_folder+f'res101_77_{split}.h5', "r")
        self.img_features = h5_file['features'] # shape [N,d,H,W]
        self.img_names = h5_file['imgnames']

        self.labels = torch.LongTensor(np.squeeze(res101['labels'][np.squeeze(att_splits[loc]-1)])) # shape [N]    
        unique_labels = np.unique(self.labels)
        self.attributes = AWA2_ATTRIBUTES
        self.classes = np.array(AWA2_CLASSES)[unique_labels-1]
        i=0
        for label in unique_labels:
            self.labels[self.labels == label] = i
            i+=1
        self.class_attributes = torch.Tensor(att_splits['att'][:, unique_labels-1]) # shape [num_attributes, num_classes]

        self.length = len(self.labels)
        assert self.length == self.img_features.shape[0]

        assert norm_type in ('std', 'L2', 'None')
        self.norm_type = norm_type
        self.norm_info = norm_info
        self.split = split
        if self.split == 'train' and self.norm_type == 'std' and self.norm_info is None:
            self.norm_info = {
                'std': self.img_features.std(0),
                'mean': self.img_features.mean(0),
            }


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = torch.Tensor(self.img_features[idx,...]) # shape [C,H,W]
        # normalize
        if self.norm_type == 'std':
            std = self.norm_info['std'].unsqueeze(0)
            std[std == 0] = 1
            mean = self.norm_info['mean'].unsqueeze(0)
            img = (img - mean) / std
        elif self.norm_type == 'L2':
            img = normalizeFeaturesL2(img.permute(1,2,0).view(49,2048)).view(7,7,2048).permute(2,0,1)
        label = self.labels[idx]
        class_attributes = self.class_attributes[:,label]
        return {'img': img, 'label': label, 'class_attributes': class_attributes}

    def get_img_path(self, idx):
        img_name = self.img_names[idx]
        class_name = img_name.split('_')[0]
        return f"/mnt/data/Animals_with_Attributes2/JPEGImages/{class_name}/{img_name}"

    def get_num_attributes(self):
        return self.class_attributes.shape[0]

    def get_img_feature_size(self):
        return self.img_features.shape[1]
