## Masked Autoencoder for Distribution Estimation
PyTorch implementation of the Masked Autoencoder for Distribution Estimation (MADE) [1]. The implemented model supports random ordering of the inputs for order-agnostic training. 

We are able to reproduce the results of [1] on MNIST in terms of negative log-likelihood and sample quality using a single-layer autoencoder with 500 hidden units. 

The animation below shows generated digits (randomly sampled using a fixed seed) from the model during training for $K=0,1,...,75$ epochs using the natural ordering of the inputs to the network:

<p align="center">
  <img src ="https://github.com/e-hulten/made/blob/master/gifs/model_500.gif">
</p>

The next figure shows reconstructed digits from the test set using the same one hidden layer network:

<p align="center">
  <img src ="https://github.com/e-hulten/made/blob/master/reconstruct/sample_75_epochs.png">
</p>

[1] https://arxiv.org/pdf/1502.03509.pdf

## TODO:
* Add functionality to train MADE with Gaussian conditionals.
* Modify plotting functions to enable plotting models trained using a random ordering of the inputs.
* Add functionality to create ensembles at test time. 
