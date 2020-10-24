import torch
import torch.utils.data
import numpy as np
import gzip
import pickle
import matplotlib.pyplot as plt
import urllib
from typing import Tuple
from torch import Tensor


class BinarisedMNIST:
    def __init__(self, download: bool = False) -> None:
        """Initialise BinarisedMNIST by loading the dataset.
        
        Args:
            download: Whether to download the dataset. 
        """
        self.path = "data/binarized_mnist.npz"
        if download:
            self._get_dataset()

        self.bmnist = np.load(self.path)

    def get_data_splits(self) -> Tuple[Tensor]:
        """Return train, validation, and test set as tensors."""
        train = torch.from_numpy(self.bmnist["train_data"])
        val = torch.from_numpy(self.bmnist["valid_data"])
        test = torch.from_numpy(self.bmnist["test_data"])
        return train, val, test

    def _get_dataset(self) -> None:
        """Internal method to download the binarized MNIST."""
        with urllib.URLopener() as downloader:
            url = "https://github.com/mgermain/MADE/releases/download/ICML2015/binarized_mnist.npz"
            dowloader.retrieve(url, self.path)
        print("Binarized MNIST successfully downloaded!")


class MNIST:
    """ 
    This is a version of: https://github.com/gpapamak/maf/blob/master/datasets/mnist.py, 
    adapted to work with Python 3.x and PyTorch. 
    """

    class Dataset:
        def __init__(self, data, logit, dequantize, rng):
            self.alpha = 1e-6
            x = (
                self._dequantize(data[0], rng) if dequantize else data[0]
            )  # dequantize pixels
            self.x = self._logit_transform(x) if logit else x  # logit
            self.N = self.x.shape[0]  # number of datapoints

        @staticmethod
        def _dequantize(x, rng):
            """
            Adds noise to pixels to dequantize them.
            """
            return x + rng.rand(*x.shape) / 256.0

        def _logit_transform(self, x):
            """
            Transforms pixel values with logit to be unconstrained.
            """
            x = self.alpha + (1 - 2 * self.alpha) * x
            return np.log(x / (1.0 - x))

    def __init__(self, logit=True, dequantize=True):
        root = "../maf/data/maf_data/"
        # load dataset
        with gzip.open(root + "mnist/mnist.pkl.gz", "rb") as f:
            train, val, test = pickle.load(f, encoding="latin1")

        rng = np.random.RandomState(42)
        self.train = self.Dataset(train, logit, dequantize, rng)
        self.val = self.Dataset(val, logit, dequantize, rng)
        self.test = self.Dataset(test, logit, dequantize, rng)

        self.n_dims = self.train.x.shape[1]
        self.image_size = [int(np.sqrt(self.n_dims))] * 2

    def get_data_splits(self):
        return (
            torch.from_numpy(self.train.x),
            torch.from_numpy(self.val.x),
            torch.from_numpy(self.test.x),
        )

    def show_pixel_histograms(self, split, pixel=None):
        """
        Shows the histogram of pixel values, or of a specific pixel if given.
        """

        data_split = getattr(self, split, None)
        if not data_split:
            raise ValueError("Invalid data split")
        if not pixel:
            data = data_split.x.flatten()
        else:
            row, col = pixel
            idx = row * self.image_size[0] + col
            data = data_split.x[:, idx]

        n_bins = int(np.sqrt(data_split.N))
        fig, ax = plt.subplots(1, 1)
        ax.hist(data, n_bins, density=True, color="lightblue")
        ax.set_yticklabels("")
        ax.set_yticks([])
        plt.show()
