# Anirudh Sathish 

import os
import glob
import torch
import numpy as np
from preprocess_transform import *
from torch.utils.data import Dataset


class EEGDataLoader(Dataset):

    def __init__(self, fold, set='train'):

        self.set = set
        self.fold = fold

        self.sr = 100        
        self.dset_cfg = {
        "name": "Sleep-EDF-2018",
        "eeg_channel": "Fpz-Cz",
        "num_splits": 10,
        "seq_len": 1,
        "target_idx": 0,
        "root_dir": "./"
    }
        self.root_dir = "./"
        self.dset_name = "Sleep-EDF-2018"
        self.num_splits = 10
        self.eeg_channel = "Fpz-Cz"
        
        self.seq_len = 1
        self.target_idx = 0

        self.training_mode = "pretrain"
        self.dataset_path = "dataset/sleep-edf-2018/npz"
        #self.dataset_path = os.path.join(self.root_dir, 'dset', self.dset_name, 'npz')
        self.inputs, self.labels, self.epochs = self.split_dataset()
        
        if self.training_mode == 'pretrain':
            self.transform = Compose(
                transforms=[
                    RandomAmplitudeScale(),
                    RandomTimeShift(),
                    RandomDCShift(),
                    RandomZeroMasking(),
                    RandomAdditiveGaussianNoise(),
                    RandomBandStopFilter(),
                ]
            )
            self.two_transform = TwoTransform(self.transform)
        
    def __len__(self):
        return len(self.epochs)

    def __getitem__(self, idx):

        n_sample = 30 * self.sr * self.seq_len
        file_idx, idx, seq_len = self.epochs[idx]
        inputs = self.inputs[file_idx][idx:idx+seq_len]

        if self.set == 'train':
            if self.training_mode == 'pretrain':
                assert seq_len == 1
                input_a, input_b = self.two_transform(inputs)
                input_a = torch.from_numpy(input_a).float()
                input_b = torch.from_numpy(input_b).float()
                inputs = [input_a, input_b]
            elif self.training_mode in ['scratch', 'fullyfinetune', 'freezefinetune']:
                inputs = inputs.reshape(1, n_sample)
                inputs = torch.from_numpy(inputs).float()
            else:
                raise NotImplementedError
        else:
            if not self.training_mode == 'pretrain':
                inputs = inputs.reshape(1, n_sample)
            inputs = torch.from_numpy(inputs).float()
        
        labels = self.labels[file_idx][idx:idx+seq_len]
        labels = torch.from_numpy(labels).long()
        labels = labels[self.target_idx]
        
        return inputs, labels

    def split_dataset(self):

        file_idx = 0
        inputs, labels, epochs = [], [], []
        data_root = os.path.join(self.dataset_path, self.eeg_channel)
        data_fname_list = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(data_root, '*.npz')))]
        data_fname_dict = {'train': [], 'test': [], 'val': []}
        split_idx_list = np.load(os.path.join('./split_idx', 'idx_{}.npy'.format(self.dset_name)), allow_pickle=True)

        assert len(split_idx_list) == self.num_splits
    
                    
        
        for i in range(len(data_fname_list)):
            subject_idx = int(data_fname_list[i][3:5])
            if subject_idx in split_idx_list[self.fold - 1][self.set]:
                data_fname_dict[self.set].append(data_fname_list[i])
    
        for data_fname in data_fname_dict[self.set]:
            npz_file = np.load(os.path.join(data_root, data_fname))
            inputs.append(npz_file['x'])
            labels.append(npz_file['y'])
            seq_len = self.seq_len
            for i in range(len(npz_file['y']) - seq_len + 1):
                epochs.append([file_idx, i, seq_len])
            file_idx += 1
        
        return inputs, labels, epochs

