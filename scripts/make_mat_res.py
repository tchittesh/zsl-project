from scipy import io
import os
from pathlib import Path
import torch
import numpy as np
from tqdm.notebook import tqdm
from tqdm import tqdm
import hdf5storage
import pickle
import h5py

# pickle.dump(res_mat, open("res101_77.pkl", "wb"))

# load_pickl = pickle.load(open("res101_77.pkl", "rb"))

# hf = h5py.File('res101_77.h5', 'w')
# hf.create_dataset('features', data=load_pickl['features'])
# hf.create_dataset('labels', data=load_pickl['labels'])
# # hf.create_dataset('image_files', data=res_mat['image_files'])
# hf.close()

# hdf5storage.savemat('res101_77_py.mat', res_mat, format='7.3', store_python_metadata=True)
# io.savemat(, res_mat, do_compression=True)


import pickle
from scipy import io
import numpy as np
import h5py

att_splits = io.loadmat('zsl-project/xlsa17/data/AWA2/att_splits.mat')
res_mat = io.loadmat('zsl-project/xlsa17/data/AWA2/res101.mat')

res_mat_filenames = [str(i).split('/')[-1].split('.jpg')[0] for i in res_mat['image_files']]

result = list(Path("resnet_feats").rglob("*.[pP][tT]"))
new_filenames = [str(j).split('/')[-1].split('.pt')[0] for j in result]
cnt = 0
assert len(res_mat_filenames) == len(new_filenames)
dicty = {}
for i in tqdm(result):
    key_name = str(i).split('/')[-1].split('.pt')[0]
    dicty[key_name] = torch.load(i, map_location=torch.device('cpu')).detach().numpy()

res77_np = np.stack([dicty[i] for i in tqdm(res_mat_filenames)], axis=-1)
loc_dict = {'train': 'train_loc', 'val': 'val_loc', 'test': 'test_unseen_loc'}
for split, loc in loc_dict.items():
    print('processing', split)
    hf = h5py.File(f'res101_77_{split}_eval.h5', 'w')
    hf.create_dataset('features',
                      data=res77_np[:, :, :, np.squeeze(att_splits[loc] - 1)].transpose(3, 0, 1, 2))
    hf.close()
