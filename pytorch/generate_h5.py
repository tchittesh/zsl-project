import pickle
from scipy import io
import numpy as np
import h5py

att_splits=io.loadmat('../xlsa17/data/AWA2/att_splits.mat')
load_pickl = pickle.load(open("../xlsa17/data/AWA2/res101_77.pkl", "rb"))
loc_dict = {'train': 'train_loc', 'val': 'val_loc', 'test': 'test_unseen_loc'}

for split, loc in loc_dict.items():
    print('processing', split)
    hf = h5py.File(f'../xlsa17/data/AWA2/res101_77_{split}.h5', 'w')
    hf.create_dataset('features', data=load_pickl['features'][:, :, :, np.squeeze(att_splits[loc]-1)].transpose(3,0,1,2))
    hf.close()
