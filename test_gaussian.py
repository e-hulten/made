import torch

from model import MADE
from utils_gaussian import test_gaussian

n_in = 784
hidden_dims = [1024]
lr = 1e-3  # try 1e-4 !!!
epochs = 200
seed = 876
random_order = False
# -------------------------------

model = MADE(n_in, hidden_dims, random_order=False, seed=seed, gaussian=True)

checkpoint = torch.load("models/model_gaussian_1024.pt")
model.load_state_dict(checkpoint["model_state_dict"])

test_gaussian(model, 0)
