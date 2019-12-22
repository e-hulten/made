import torch
from torch.nn import functional as F
from torchvision.utils import save_image
from torch.distributions import Bernoulli
import numpy as np
import os

from model import MADE, MADECompanion
from utils import test, sample_digits


# --------- parameters ----------
n_in = 784
hidden_dims = [500]
seed = 7
# -------------------------------

model = MADE(n_in, hidden_dims, random_order=False, seed=seed, gaussian=False)

string = ""
for i in range(len(hidden_dims)):
    string += "_" + str(hidden_dims[i])

checkpoint = torch.load("models/model" + string + ".pt")
model.load_state_dict(checkpoint["model_state_dict"])
tot_epochs = checkpoint["epoch"]

test(model, tot_epochs, plot=True)
sample_digits(model, tot_epochs, test=True)

