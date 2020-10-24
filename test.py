import torch
from made import MADE
from data import MNIST
from utils import test, sample_digits, sample_digits_half, sample_best
from test import test_model_gaussian


# Get datasets and train loaders.
mnist = MNIST()
_, _, test = mnist.get_data_splits()
test_loader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=True)

# --------- parameters ----------
n_in = 784
hidden_dims = [8000]
seed = 19
random_order = False
# -------------------------------

model = MADE(n_in, hidden_dims, random_order=random_order, seed=seed, gaussian=False)

string = "_".join([str(h) for h in hidden_dims])

checkpoint = torch.load("model_saves/model" + string + ".pt")
model.load_state_dict(checkpoint["model_state_dict"])
tot_epochs = checkpoint["epoch"]

# sample_digits(model, tot_epochs, random_order=random_order, test=True)
# sample_best(model, tot_epochs)
# sample_digits_half(model, tot_epochs, random_order=random_order, test=True)
test(model, tot_epochs, plot=True)
