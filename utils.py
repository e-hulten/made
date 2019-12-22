import torch
from torch.nn import functional as F
from torch.distributions import Bernoulli
from torchvision.utils import save_image
import os

from data import train_loader, test_loader


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


def test(model, tot_epochs, plot=False):
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

        test_loss /= len(test_loader.dataset)
        print("Test loss: {:.4f}".format(test_loss))


def sample_digits(model, epoch, seed=None, test=False):
    model.eval()
    n_samples = 80
    if seed is not None:
        torch.manual_seed(seed)
    samples = torch.zeros(n_samples, 784)
    # sample the first dimension of each vector
    samples[:, 0] = torch.round(torch.rand(n_samples))
    for dim in range(0, 784):
        output = model(samples)
        bernoulli = torch.distributions.Bernoulli(output[:, dim])
        sample_output = bernoulli.sample()
        samples[:, dim] = sample_output

    samples = samples.cpu().view(n_samples, 1, 28, 28)
    if not os.path.exists("gif_results"):
        os.makedirs("gif_results")
    if test is False:
        save_path = "gif_results/samples_" + str(epoch) + ".png"
    else:
        save_path = "results/samples_" + str(epoch) + ".png"

    save_image(samples, save_path, nrow=10)


def sample_best(model, epoch):
    n_samples = 10000
    num_best = 80
    samples = torch.zeros(n_samples, 784)
    log_probs = torch.zeros(n_samples)

    samples[:, 0] = torch.rand(n_samples)
    for dim in range(0, 784):
        output = model(samples)
        bernoulli = torch.distributions.Bernoulli(output[:, dim])
        sample_output = bernoulli.sample()
        samples[:, dim] = sample_output

    _, idx = log_probs.topk(num_best)
    best = samples[idx, :]
    sample = best.view(num_best, 1, 28, 28)
    save_path = "results/sample_best_" + str(epoch) + ".png"
    save_image(sample, save_path, nrow=10)


def plot_comparison(batch, x_hat, tot_epochs, num_samples=10):
    comparison = torch.cat(
        [
            batch.view(128, 1, 28, 28)[:num_samples],
            x_hat.view(128, 1, 28, 28)[:num_samples],
            batch.view(128, 1, 28, 28)[num_samples : 2 * num_samples],
            x_hat.view(128, 1, 28, 28)[num_samples : 2 * num_samples],
            batch.view(128, 1, 28, 28)[2 * num_samples : 3 * num_samples],
            x_hat.view(128, 1, 28, 28)[2 * num_samples : 3 * num_samples],
        ]
    )

    if not os.path.exists("reconstruct"):
        os.makedirs("reconstruct")
    save_path = "reconstruct/sample_" + str(tot_epochs) + "_epochs.png"
    save_image(comparison, save_path, nrow=num_samples, pad_value=1, padding=1)

