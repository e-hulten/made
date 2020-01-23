import torch
from torch.nn import functional as F
from torch.distributions import Bernoulli
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
import math

from mnist import (
    train_loader as train_loader_mnist,
    val_loader as val_loader_mnist,
    test_loader as test_loader_mnist,
)


def train_one_epoch_gaussian(model, epoch, optimizer, scheduler=None):
    model.train()
    train_loss = 0
    for idx, batch in enumerate(train_loader_mnist):
        optimizer.zero_grad()

        batch = batch.reshape(-1, 28 * 28)
        out = model.forward(batch.float())
        mu, alpha = torch.chunk(out.clone(), 2, dim=1)
        u = (batch - mu) / torch.exp(alpha)

        negloglik_loss = 0.5 * (u * u).sum(dim=1)
        negloglik_loss += (
            0.5 * batch.shape[1] * torch.log(torch.from_numpy(np.array(2 * np.pi)))
        )
        negloglik_loss += alpha.sum(dim=1)
        negloglik_loss = torch.sum(negloglik_loss)

        negloglik_loss.backward()
        train_loss += negloglik_loss.item()
        optimizer.step()
        if scheduler is not None:
            scheduler.step(epoch)

    avg_loss = train_loss / len(train_loader_mnist.dataset)
    print("Epoch: {} Average loss: {:.5f}".format(epoch, avg_loss))
    return avg_loss


def val_gaussian(model, tot_epochs):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for idx, batch in enumerate(val_loader_mnist):
            batch = batch.view(-1, 784)
            out = model.forward(batch.float())
            mu, alpha = torch.chunk(out.clone(), 2, dim=1)
            u = (batch - mu) / torch.exp(alpha)

            negloglik_loss = 0.5 * (u * u).sum(dim=1)
            negloglik_loss += (
                0.5 * batch.shape[1] * torch.log(torch.from_numpy(np.array(2 * np.pi)))
            )
            negloglik_loss += alpha.sum(dim=1)
            negloglik_loss = torch.sum(negloglik_loss)

            val_loss += negloglik_loss

        val_loss /= len(val_loader_mnist.dataset)
        print("Validation loss: {:.4f}".format(val_loss))
    return val_loss


def test_gaussian(model, tot_epochs, plot=False):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for idx, batch in enumerate(test_loader_mnist):
            batch = batch.view(-1, 784)
            out = model.forward(batch.float())
            mu, alpha = torch.chunk(out.clone(), 2, dim=1)
            u = (batch - mu) / torch.exp(alpha)

            negloglik_loss = 0.5 * (u * u).sum(dim=1)
            negloglik_loss += (
                0.5 * batch.shape[1] * torch.log(torch.from_numpy(np.array(2 * np.pi)))
            )
            negloglik_loss += alpha.sum(dim=1)
            negloglik_loss = torch.sum(negloglik_loss)

            test_loss += negloglik_loss

        test_loss /= len(test_loader_mnist.dataset)
        print("Test loss: {:.4f}".format(test_loss))


def sample_digits_gaussian(model, epoch, random_order=False, seed=None, test=False):
    model.eval()
    n_samples = 80
    if seed is not None:
        torch.manual_seed(seed)
    if random_order is True:
        np.random.seed(seed)
        order = np.random.permutation(784)
    else:
        order = np.arange(784)

    samples = torch.zeros(n_samples, 784)
    # sample the first dimension of each vector
    samples[:, order[0]] = torch.rand(n_samples)
    eps = samples.clone().normal_(0, 1)
    for _, dim in enumerate(order):
        out = model(samples)
        mu, alpha = torch.chunk(out.clone(), 2, dim=1)

        x = mu[:, dim] + torch.exp(alpha[:, dim]) * eps[:, dim]

        samples[:, dim] = x

    samples = (torch.sigmoid(samples) - 1e-6) / (1 - 2e-6)

    samples = samples.detach().cpu().view(n_samples, 28, 28)

    fig, axes = plt.subplots(ncols=10, nrows=8)
    ax = axes.ravel()
    for i in range(80):
        ax[i].imshow(
            np.transpose(samples[i], (0, 1)), cmap="gray", interpolation="none"
        )
        ax[i].axis("off")
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
        ax[i].set_frame_on(False)

    if not os.path.exists("gif_results"):
        os.makedirs("gif_results")
    if test is False:
        save_path = "gif_results/samples_gaussian_" + str(epoch) + ".pdf"
    else:
        save_path = "results/samples_gaussian_" + str(epoch) + ".pdf"

    fig.subplots_adjust(wspace=-0.35, hspace=0.065)
    plt.gca().set_axis_off()
    plt.savefig(
        save_path, dpi=300, bbox_inches="tight", pad_inches=0,
    )
    plt.close()


def plot_losses(epochs, train_losses, val_losses, title=None):
    sns.set(style="white")
    df = pd.DataFrame({"Epoch": epochs, "Train": train_losses, "Val": val_losses})
    df["Train"] = df["Train"].astype(float)
    df["Val"] = df["Val"].astype(float)
    print(df)
    fig, axes = plt.subplots(
        ncols=1, nrows=1, figsize=[13, 6], sharey=True, sharex=True, dpi=400
    )
    axes = sns.lineplot(
        x="Epoch",
        y="value",
        hue="",
        data=pd.melt(df, ["Epoch"]).rename(columns={"variable": ""}),
        palette="tab10",
    )
    axes.set_ylabel("Loss")
    axes.legend(
        frameon=False,
        prop={"size": 14},
        fancybox=False,
        handletextpad=0.5,
        handlelength=1,
    )
    axes.set_ylim(1350, 2100)
    axes.set_title(title) if title is not None else axes.set_title(None)

    save_path = "results/train_plots" + str(epochs[-1]) + ".pdf"
    plt.savefig(
        save_path, dpi=300, bbox_inches="tight", pad_inches=0,
    )
    plt.close()
