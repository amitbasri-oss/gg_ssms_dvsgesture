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
            val_idx = 0
    ):
        transform = tonic.transforms.Compose(
            [tonic.transforms.ToFrame(sensor_size=tonic.datasets.DVSGesture.sensor_size, n_event_bins=self.n_bins),
             tonic.transforms.NumpyAsType(np.float32)])
        # transform = tonic.transforms.NumpyAsType(np.float32)
        if name == "val":
            self.y = tonic.datasets.DVSGesture(save_to='/mnt/c/Users/Admin/Documents', train=False, transform=transform)
        else:
            self.y = tonic.datasets.DVSGesture(save_to='/mnt/c/Users/Admin/Documents', train=True, transform=transform)
        self.name = name
        self.val_idx = val_idx

    def __len__(self):
        if self.name == "val":
            return len(self.y)
        return (len(self.y)) * (len(self.y[0][0]) - 1)

    def __repr__(self):
        return self.__class__.__name__

    def __getitem__(self, index):
        if name == "val":
            return self.y[index][0][self.val_idx], self.y[index][1]
        l = index % (len(self.y[0][0]) - 1)
        if l >= self.val_idx:
            l += 1
        i = index // (len(self.y[0][0]) - 1)
        return self.y[i][0][l], self.y[i][1]
