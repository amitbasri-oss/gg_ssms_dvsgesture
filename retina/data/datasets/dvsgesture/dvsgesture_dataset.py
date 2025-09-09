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
        self.val_idx = val_idx

    def __len__(self):
        if self.name == "val":
            return 20
        return len(self.y) - 20

    def __repr__(self):
        return self.__class__.__name__

    def __getitem__(self, index):
        if self.name == "val":
            return self.y[index]
        return self.y[index+20]
