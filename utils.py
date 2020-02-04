import torch
from torch.nn import functional as F
from torch.distributions import Bernoulli
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
import numpy as np
import math

from data import train_loader, val_loader, test_loader
from mnist import train_loader, test_loader as train_loader_mnist, test_loader_mnist


def train_one_epoch(model, epoch, optimizer, scheduler=None):
    model.train()
    train_loss = 0
    for idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        batch = batch.reshape(-1, 28 * 28)
        x_hat = model.forward(batch)

        binary_loss = F.binary_cross_entropy(
            x_hat, batch.reshape(-1, 28 * 28), reduction="sum"
        )

        binary_loss.backward()
        loss = binary_loss.item()
        train_loss += loss
        optimizer.step()
        if scheduler is not None:
            scheduler.step(epoch)

    avg_loss = train_loss / len(train_loader.dataset)
    print("Epoch: {} Average loss: {:.5f}".format(epoch, avg_loss))


def val(model, tot_epochs, plot=False):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            batch = batch.view(-1, 784)
            x_hat = model(batch)
            binary_loss = F.binary_cross_entropy(
                x_hat, batch.reshape(-1, 784), reduction="sum"
            )
            val_loss += binary_loss
            if plot is True:
                if idx == 50:
                    plot_comparison(batch, x_hat, tot_epochs, num_samples=10)

        val_loss /= len(val_loader.dataset)
        print("Validation loss: {:.4f}".format(val_loss))


def test(model, tot_epochs):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            batch = batch.view(-1, 784)
            x_hat = model(batch)
            binary_loss = F.binary_cross_entropy(
                x_hat, batch.reshape(-1, 784), reduction="sum"
            )
            test_loss += binary_loss
            if plot is True:
                if idx == 50:
                    plot_comparison(batch, x_hat, tot_epochs, num_samples=10)
                    plot_comparison_one_row(
                        batch, x_hat, tot_epochs,
                    )

        test_loss /= len(test_loader.dataset)
        print("Test loss: {:.4f}".format(test_loss))


def sample_digits(model, epoch, random_order=False, seed=None, test=False):
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
    samples[:, order[0]] = torch.round(torch.rand(n_samples))
    for _, dim in enumerate(order):
        output = model(samples)
        bernoulli = torch.distributions.Bernoulli(output[:, dim])
        sample_output = bernoulli.sample()
        samples[:, dim] = sample_output
    samples = samples.cpu().view(n_samples, 28, 28)

    fig, axes = plt.subplots(ncols=10, nrows=8)
    ax = axes.ravel()
    for i in range(80):
        ax[i].imshow(np.transpose(samples[i], (0, 1)), cmap="gray")
        ax[i].axis("off")
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
        ax[i].set_frame_on(False)

    if not os.path.exists("gif_results"):
        os.makedirs("gif_results")
    if test is False:
        save_path = "gif_results/samples_" + str(epoch) + ".pdf"
    else:
        save_path = "results/samples_" + str(epoch) + ".pdf"

    fig.subplots_adjust(wspace=-0.35, hspace=0.065)
    plt.gca().set_axis_off()
    plt.savefig(
        save_path, dpi=300, bbox_inches="tight", pad_inches=0,
    )
    plt.close()


def sample_best(model, epoch):
    n_samples = 1000
    num_best = 15 * 15
    samples = torch.zeros(n_samples, 784)
    log_probs = torch.zeros(n_samples)

    samples[:, 0] = torch.rand(n_samples)
    for dim in range(0, 784):
        output = model(samples)
        bernoulli = torch.distributions.Bernoulli(output[:, dim])
        sample_output = bernoulli.sample()
        samples[:, dim] = sample_output
        log_probs += bernoulli.log_prob(samples[:, dim])

    _, idx = log_probs.topk(num_best)
    best = samples[idx, :]
    print(log_probs[idx])
    sample = best.view(num_best, 1, 28, 28)
    save_path = "results/sample_best_" + str(epoch) + ".png"
    save_image(sample, save_path, nrow=15)


def plot_comparison(batch, x_hat, tot_epochs, num_samples=10):
    comparison = torch.cat(
        [batch.view(128, 28, 28)[:num_samples], x_hat.view(128, 28, 28)[:num_samples]]
    )
    fig, axes = plt.subplots(ncols=num_samples, nrows=3, figsize=(10, 3))
    ax = axes.ravel()
    j = 0  # global plot counter
    for i in range(num_samples):
        ax[i].imshow(np.transpose(comparison[i], (0, 1)), cmap="gray")
        ax[i].axis("on")
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
        ax[i].xaxis.set_ticks_position("none")
        ax[i].yaxis.set_ticks_position("none")
        ax[i].set_frame_on(False)
        j += 1
    for i in range(num_samples, 2 * num_samples):
        ax[i].imshow(np.transpose(comparison[i], (0, 1)), cmap="Blues")

        ax[i].axis("on")
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
        ax[i].xaxis.set_ticks_position("none")
        ax[i].yaxis.set_ticks_position("none")
        j += 1

    samples = torch.zeros(num_samples, 784)
    for dim in range(784):
        bernoulli = torch.distributions.Bernoulli(x_hat[:num_samples, dim])
        sample_output = bernoulli.sample()
        samples[:, dim] = sample_output

    samples = samples.cpu().view(num_samples, 28, 28)

    for i in range(0, num_samples):
        ax[j].imshow(np.transpose(samples[i], (0, 1)), cmap="gray")
        ax[i].axis("on")
        ax[j].set_xticklabels([])
        ax[j].set_yticklabels([])
        ax[j].xaxis.set_ticks_position("none")
        ax[j].yaxis.set_ticks_position("none")
        j += 1

    fig.subplots_adjust(wspace=0.04, hspace=0.04)
    plt.gca().set_axis_off()

    if not os.path.exists("reconstruct"):
        os.makedirs("reconstruct")
    save_path = "reconstruct/reconstruct_" + str(tot_epochs) + "_epochs.pdf"
    axes[0, 0].set(ylabel="$\mathbf{x}$")
    axes[1, 0].set(ylabel="$p(x_i = 1\mid\mathbf{x}_{{<i}})$")
    axes[2, 0].set(ylabel="$\mathbf{x}'$")
    plt.savefig(
        save_path, dpi=300, bbox_inches="tight", pad_inches=0,
    )


def sample_digits_half(model, epoch, random_order=False, seed=None, test=False):
    model.eval()
    n_samples = 80
    if seed is not None:
        torch.manual_seed(seed)
    if random_order is True:
        np.random.seed(seed)
        order = np.random.permutation(784)
    else:
        order = np.arange(784)

    fig, axes = plt.subplots(ncols=3, nrows=1)
    samples = torch.zeros(1, 784)
    batch = next(iter(test_loader))
    img = batch[0].view(28, 28)

    fig, axes = plt.subplots(ncols=3, nrows=1)
    axes[0].imshow(img, cmap="gray")
    axes[0].axis("off")
    axes[0].set_xticklabels([])
    axes[0].set_yticklabels([])
    axes[0].set_frame_on(False)

    img[14:, :] = 0.5
    img = img.reshape(1, 784)

    # sample the first dimension of each vector
    samples[0 : int(784 / 2)] = img[0 : int(784 / 2)]
    samples_plt = samples.cpu().view(28, 28)

    axes[1].imshow(samples_plt, cmap="gray")
    axes[1].axis("off")
    axes[1].set_xticklabels([])
    axes[1].set_yticklabels([])
    axes[1].set_frame_on(False)
    for dim in range(int(784 / 2), 784):
        output = model(samples)
        bernoulli = torch.distributions.Bernoulli(output[:, dim])
        sample_output = bernoulli.sample()
        samples[:, dim] = sample_output
    samples = samples.cpu().view(28, 28)

    axes[2].imshow(samples, cmap="gray")
    axes[2].axis("off")
    axes[2].set_xticklabels([])
    axes[2].set_yticklabels([])
    axes[2].set_frame_on(False)

    if not os.path.exists("gif_results"):
        os.makedirs("gif_results")
    if test is False:
        save_path = "gif_results/samples_half" + str(epoch) + ".pdf"
    else:
        save_path = "results/samples_half" + str(epoch) + ".pdf"

    fig.subplots_adjust(wspace=0.1, hspace=0.065)
    plt.gca().set_axis_off()
    plt.savefig(
        save_path, dpi=300, bbox_inches="tight", pad_inches=0,
    )


def plot_comparison_one_row(batch, x_hat, tot_epochs, num_samples=1):
    comparison = torch.cat(
        [batch.view(128, 28, 28)[:num_samples], x_hat.view(128, 28, 28)[:num_samples]]
    )
    fig, axes = plt.subplots(ncols=3, nrows=num_samples, figsize=(10, 3))
    ax = axes.ravel()
    j = 0  # global plot counter
    for i in range(num_samples):
        ax[i].imshow(np.transpose(comparison[i], (0, 1)), cmap="gray")
        ax[i].axis("on")
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
        ax[i].xaxis.set_ticks_position("none")
        ax[i].yaxis.set_ticks_position("none")
        ax[i].set_frame_on(False)
        j += 1
    for i in range(num_samples, 2 * num_samples):
        ax[i].imshow(np.transpose(comparison[i], (0, 1)), cmap="Blues")

        ax[i].axis("on")
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
        ax[i].xaxis.set_ticks_position("none")
        ax[i].yaxis.set_ticks_position("none")
        j += 1

    samples = torch.zeros(num_samples, 784)
    for dim in range(784):
        bernoulli = torch.distributions.Bernoulli(x_hat[:num_samples, dim])
        sample_output = bernoulli.sample()
        samples[:, dim] = sample_output

    samples = samples.cpu().view(num_samples, 28, 28)

    for i in range(0, num_samples):
        ax[j].imshow(np.transpose(samples[i], (0, 1)), cmap="gray")
        ax[i].axis("on")
        ax[j].set_xticklabels([])
        ax[j].set_yticklabels([])
        ax[j].xaxis.set_ticks_position("none")
        ax[j].yaxis.set_ticks_position("none")
        j += 1

    fig.subplots_adjust(wspace=0.04, hspace=0.04)
    plt.gca().set_axis_off()

    if not os.path.exists("reconstruct"):
        os.makedirs("reconstruct")
    save_path = "reconstruct/reconstruct_one_row_" + str(tot_epochs) + "_epochs.pdf"
    axes[0].set(title="$\mathbf{x}$")
    axes[1].set(title="$p(x_i = 1\mid\mathbf{x}_{{<i}})$")
    axes[2].set(title="$\mathbf{x}'$")
    plt.savefig(
        save_path, dpi=300, bbox_inches="tight", pad_inches=0,
    )
