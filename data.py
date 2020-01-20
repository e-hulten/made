import torch
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import os

# download binarised mnist: https://github.com/mgermain/MADE/releases/download/ICML2015/binarized_mnist.npz
# path to binarized_mnist.npz:
path = "./data/binarized_mnist.npz"
batch_size = 128

mnist = np.load(path)

train, val, test = mnist["train_data"], mnist["valid_data"], mnist["test_data"]
train = torch.from_numpy(train)
test = torch.from_numpy(test)

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,)
val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True,)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True,)


def plot_training_samples(batch, ncol=10, nrow=2):
    tr_samples = batch.view(128, 28, 28)

    fig, axes = plt.subplots(ncols=ncol, nrows=nrow, figsize=(15, 3))
    ax = axes.ravel()
    for i in range(ncol * nrow):
        ax[i].imshow(np.transpose(tr_samples[i], (0, 1)), cmap="gray")
        ax[i].axis("off")
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
        ax[i].set_frame_on(False)
    fig.subplots_adjust(wspace=0.05, hspace=0)
    plt.gca().set_axis_off()

    if not os.path.exists("training_samples"):
        os.makedirs("training_samples")
    save_path = "training_samples/samples.pdf"

    plt.savefig(
        save_path, dpi=300, bbox_inches="tight", pad_inches=0,
    )


batch = next(iter(train_loader))
plot_training_samples(batch)
