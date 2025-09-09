"""
This is a file script used for loading the dataloader
"""

import pdb
import torch
from PIL import Image
import numpy as np

# custom
from data.utils import load_yaml_config
from data.transforms.helper import get_transforms
from data.datasets.dvsgesture.dvsgesture_dataset import DVSGestureDataset


def get_dvsgesture_dataset(name, training_params, dataset_params):
    """
    Create and return a Dataset from the Ini30Dataset.

    Parameters:
        data_dir (str): The directory path where the dataset is located.
        batch_size (int): The batch size used in the DataLoader.
        num_bins (int): The number of bins used for transformation.
        idxs (List[int]): A list of experiment indices to include in the dataset.

    Returns:
        Dataset: The Dataset object from the Ini30Dataset.
    """

    dataset = DVSGestureDataset(
        training_params=training_params, dataset_params=dataset_params, name=name, val_idx=dataset_params["ini30_val_idx"])

    return dataset