import torch
from models import MADE
from data import MNIST
from utils import train_one_epoch, sample_digits, sample_best, test

# --------- parameters ----------
n_in = 784
hidden_dims = [1024]
lr = 1e-3
epochs = 75
seed = 19
random_order = False
batch_size = 100
# -------------------------------

mnist = MNIST()
train, val, test = mnist.get_data_splits()
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,)

model = MADE(n_in, hidden_dims, random_order=random_order, seed=seed, gaussian=False)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

for epoch in range(epochs):
    train_one_epoch(model, train_loader, epoch, optimizer, scheduler=scheduler)
    sample_digits(model, epoch, random_order=random_order, seed=seed)
    # test(model, epoch)


string = "_".join([str(h) for h in hidden_dims])
torch.save(
    {
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    },
    "./model_saves/model_" + string + ".pt",
)
