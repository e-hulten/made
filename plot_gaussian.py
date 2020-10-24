import torch
from torch.nn import functional as F
from torchvision.utils import save_image
import numpy as np
import os

from made import MADE
from utils_gaussian import (
    train_one_epoch_gaussian,
    val_gaussian,
    test_gaussian,
    sample_digits_gaussian,
    plot_losses,
)

# --------- parameters ----------
load_model = True
n_in = 784
hidden_dims = [1024]
lr = 1e-3  # try 1e-4 !!!
epochs = 200
seed = 19
random_order = False
# -------------------------------

model = MADE(n_in, hidden_dims, random_order=random_order, seed=seed, gaussian=True)


string = ""
for i in range(len(hidden_dims)):
    string += "_" + str(hidden_dims[i])

if not os.path.exists("models"):
    os.makedirs("models")

if load_model is True:
    checkpoint = torch.load("models/model_gaussian" + string + ".pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    tot_epochs = checkpoint["epoch"]
else:
    tot_epochs = 0

sample_digits_gaussian(model, tot_epochs, random_order=False, seed=seed)
