import numpy as np
import torch

from utils import normalizeFeaturesL2

class ZSLDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_name, split, norm_type='none', norm_info=None):
        assert dataset_name in ('APY', 'AWA1', 'AWA2', 'CUB', 'ImageNet', 'SUN')
        loc_dict = {'train': 'train_loc', 'val': 'val_loc', 'test': 'test_unseen_loc'}
        assert split in loc_dict
        loc = loc_dict[split]
        
        # load data
		data_folder = '../xlsa17/data/'+dataset_name+'/'
		res101 = io.loadmat(data_folder+'res101.mat')
		att_splits=io.loadmat(data_folder+'att_splits.mat')

        # filter based on split
		self.img_features = torch.Tensor(res101['features'][:, np.squeeze(att_splits[loc]-1)]).permute(1,0) # shape [N,d]
		self.labels = torch.Tensor(np.squeeze(res101['labels'][np.squeeze(att_splits[loc]-1)])) # shape [N]    
		unique_labels = np.unique(self.labels)
		i=0
		for label in unique_labels:
			self.labels[self.label == label] = i    
			i+=1
		self.class_attributes = torch.Tensor(att_splits['att'][:, unique_labels-1]) # shape [num_attributes, num_classes]

        self.length = len(self.labels)
        assert self.length == self.img_features.shape[0]

        assert norm_type in ('std', 'l2', 'none')
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
        elif norm_type == 'l2':
			self.img_features = normalizeFeaturesL2(self.img_features)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {'img': self.img_features[idx,:], 'label': self.labels[idx]}
