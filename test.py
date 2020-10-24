import torch
from torch.nn import functional as F
from torchvision.utils import save_image
from torch.distributions import Bernoulli
import numpy as np
import os

from made import MADE, MADECompanion
from utils import test, sample_digits, sample_digits_half, sample_best


# --------- parameters ----------
n_in = 784
hidden_dims = [8000]
seed = 19
random_order = False
# -------------------------------

model = MADE(n_in, hidden_dims, random_order=random_order, seed=seed, gaussian=False)

string = ""
for i in range(len(hidden_dims)):
    string += "_" + str(hidden_dims[i])

checkpoint = torch.load("models/model" + string + ".pt")
model.load_state_dict(checkpoint["model_state_dict"])
tot_epochs = checkpoint["epoch"]

# sample_digits(model, tot_epochs, random_order=random_order, test=True)
# sample_best(model, tot_epochs)
sample_digits_half(model, tot_epochs, random_order=random_order, test=True)
test(model, tot_epochs, plot=True)
