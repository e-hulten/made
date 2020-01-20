import torch
from torch.nn import functional as F
from torchvision.utils import save_image
import numpy as np
import os

from model import MADE
from utils_gaussian import (
    train_one_epoch_gaussian,
    val_gaussian,
    test_gaussian,
    sample_digits_gaussian,
)

# --------- parameters ----------
load_model = False
n_in = 784
hidden_dims = [1000]
lr = 1e-3
epochs = 60
seed = 19
random_order = False
# -------------------------------

model = MADE(n_in, hidden_dims, random_order=random_order, seed=seed, gaussian=True)
print(
    "Number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()])
)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

string = ""
for i in range(len(hidden_dims)):
    string += "_" + str(hidden_dims[i])

if load_model is True:
    checkpoint = torch.load("models/model_gaussian" + string + ".pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    tot_epochs = checkpoint["epoch"]
else:
    tot_epochs = 0
    sample_digits_gaussian(model, 0)

for epoch in range(tot_epochs + 1, tot_epochs + epochs + 1):
    train_one_epoch_gaussian(model, epoch, optimizer, scheduler=scheduler)
    sample_digits_gaussian(model, epoch, random_order=random_order, seed=seed)
    val_gaussian(model, epoch)
test_gaussian(model, epoch)

if not os.path.exists("models"):
    os.makedirs("models")
torch.save(
    {
        "epoch": tot_epochs + epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        # "scheduler_state_dict": scheduler.state_dict(),
    },
    "./models/model_gaussian" + string + ".pt",
)
