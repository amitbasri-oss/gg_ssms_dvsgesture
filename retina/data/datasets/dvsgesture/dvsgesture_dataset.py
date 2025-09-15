"""
This is a file script used for loading the dataset
"""

import torch
import numpy as np
import tonic

class DVSGestureDataset:
    def __init__(
            self,
            training_params = None,
            dataset_params = None,
            name = "val",
            val_idx = range(20)
    ):
        n_bins = 20
        if dataset_params is not None:
            n_bins = dataset_params["num_bins"]
        transform = tonic.transforms.Compose(
            [tonic.transforms.ToFrame(sensor_size=tonic.datasets.DVSGesture.sensor_size, n_event_bins=n_bins),
             tonic.transforms.NumpyAsType(np.float32)])
        # transform = tonic.transforms.NumpyAsType(np.float32)
        if name == "val":
            self.y = tonic.datasets.DVSGesture(save_to='/mnt/c/Users/Admin/Documents', train=False, transform=transform)
        else:
            self.y = tonic.datasets.DVSGesture(save_to='/mnt/c/Users/Admin/Documents', train=True, transform=transform)
        self.name = name
        self.idx_lst = []
        for i in range(len(self.y)):
            if (name == "val") == (i in val_idx):
                self.idx_lst.append(i)

    def __len__(self):
        return len(self.idx_lst)

    def __repr__(self):
        return self.__class__.__name__

    def __getitem__(self, index):
        return self.y[self.idx_lst[index]]
