import torch
import numpy as np
import math
from torch.nn import functional as F
from utils.plot import plot_comparison

#####################################################################################
# Bernoulli MADE
#####################################################################################
def train_epoch(model, train_loader, epoch, optimizer, scheduler=None):
    model.train()
    train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()

        batch = batch.reshape(-1, 28 * 28).float()
        x_hat = model.forward(batch)

        binary_loss = F.binary_cross_entropy(
            x_hat, batch.reshape(-1, 28 * 28), reduction="sum"
        )

        binary_loss.backward()
        loss = binary_loss.item()
        train_loss += loss
        optimizer.step()
        if scheduler:
            scheduler.step(epoch)

    avg_loss = train_loss / len(train_loader.dataset)
    print("(Epoch {}) Average loss: {:.4f}".format(str(epoch).zfill(3), avg_loss))


def validate_epoch(model, val_loader, tot_epochs, plot=False):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            batch = batch.view(-1, 784)
            x_hat = model(batch)
            binary_loss = F.binary_cross_entropy(
                x_hat, batch.reshape(-1, 784), reduction="sum"
            )
            val_loss += binary_loss
            if plot and idx == 50:
                plot_comparison(batch, x_hat, tot_epochs, num_samples=10)

        val_loss /= len(val_loader.dataset)
        print("{}Validation loss: {:.4f}".format(" " * 12, val_loss))


#####################################################################################
# Gaussian MADE
#####################################################################################
def train_epoch_gaussian(model, train_loader, epoch, optimizer):
    model.train()
    train_loss = 0
    for batch in train_loader:
        batch = batch.reshape(-1, 28 * 28)
        out = model.forward(batch.float())
        mu, alpha = torch.chunk(out, 2, dim=1)
        u = (batch - mu) * torch.exp(-alpha)

        negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
        negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)
        negloglik_loss += torch.sum(alpha, dim=1)
        negloglik_loss = torch.mean(negloglik_loss)

        optimizer.zero_grad()
        negloglik_loss.backward()
        train_loss += negloglik_loss.item()
        optimizer.step()

    avg_loss = train_loss / len(train_loader)
    print("Epoch: {} Average loss: {:.5f}".format(epoch, avg_loss))


def validate_epoch_gaussian(model, val_loader, tot_epochs):
    model.eval()
    val_loss = 0
    val_loss = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.reshape(-1, 28 * 28)
            out = model.forward(batch.float())

            mu, alpha = torch.chunk(out.clone(), 2, dim=1)
            u = (batch - mu) * torch.exp(-alpha)

            negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
            negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)
            negloglik_loss += torch.sum(alpha, dim=1)

            val_loss.extend(negloglik_loss)

    print("Validation loss: {:.4f}".format(np.mean(val_loss)))
    return np.mean(val_loss)
