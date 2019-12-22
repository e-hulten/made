import torch
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import os

# download binarised mnist: https://github.com/mgermain/MADE/releases/download/ICML2015/binarized_mnist.npz
# path to binarized_mnist.npz:
path = "./data/binarized_mnist.npz"
batch_size = 128

mnist = np.load(path)

train, test = mnist["train_data"], mnist["test_data"]
train = torch.from_numpy(train)
test = torch.from_numpy(test)

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True,)


def plot_training_samples(batch, num_samples=10):
    tr_samples = torch.cat(
        [
            batch.view(128, 1, 28, 28)[:num_samples],
            batch.view(128, 1, 28, 28)[num_samples : 2 * num_samples],
        ]
    )

    if not os.path.exists("training_samples"):
        os.makedirs("training_samples")
    save_path = "training_samples/samples.pdf"
    save_image(tr_samples, save_path, nrow=num_samples, pad_value=1, padding=1)


batch = next(iter(train_loader))
plot_training_samples(batch)
