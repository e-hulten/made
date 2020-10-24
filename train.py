import torch
from models import MADE
from data import BinarisedMNIST
from train import train_epoch, validate_epoch
from utils.plot import plot_comparison, sample_digits

epochs = 5
save_model = False
seed = 290713

# Get datasets and train loaders.
bmnist = BinarisedMNIST()
train, val, _ = bmnist.get_data_splits()
train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size=128, shuffle=True)

# Define model, optimizer, and scheduler.
model = MADE(n_in=784, hidden_dims=[1024], random_order=False, seed=seed)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# Training loop.
for epoch in range(1, epochs + 1):
    train_epoch(model, train_loader, epoch, optimizer, scheduler=scheduler)
    sample_digits(model, epoch, seed=seed)
    validate_epoch(model, val_loader, epoch)

if save_model:
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
