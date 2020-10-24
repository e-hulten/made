import torch
import numpy as np
import os
from models import MADE
from data import MNIST
from train import (
    train_epoch_gaussian,
    validate_epoch_gaussian,
)
from test import test_model_gaussian
from utils.plot import sample_digits_gaussian


# --------- parameters ----------
load_model = False
n_in = 784
hidden_dims = [1024]
lr = 1e-3  # try 1e-4 !!!
epochs = 200
seed = 876
random_order = False
save_model = True
# -------------------------------

# Get datasets and train loaders.
mnist = MNIST()
train, val, _ = mnist.get_data_splits()
train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size=128, shuffle=True)

model = MADE(n_in, hidden_dims, random_order=False, seed=seed, gaussian=True)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

# Early stopping
i = 0
max_loss = np.inf

for epoch in range(1, epochs + 1):
    train_epoch_gaussian(model, train_loader, epoch, optimizer)
    # sample_digits_gaussian(model, epoch, random_order=random_order, seed=seed)
    val_loss = validate_epoch_gaussian(model, val_loader, epoch)
    if val_loss < max_loss:
        max_loss = val_loss
        i = 0
    else:
        i += 1
    if i > 20:
        break
    print("Early stopping counter:", i)

if save_model:
    string = "_".join([str(h) for h in hidden_dims])
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        "./model_saves/model_gaussian_" + string + ".pt",
    )
