## Masked Autoencoder for Distribution Estimation
PyTorch implementation of the Masked Autoencoder for Distribution Estimation (MADE) [1]. The implemented model supports random ordering of the inputs for order-agnostic training. 

We are able to reproduce the results of [1] on binarised MNIST in terms of negative log-likelihood for one hidden layer MADEs with 500 and 8000 hidden units, using random and natural ordering of the inputs, respectively. We are also able to reproduce the results of the two hidden layer MADE with 8000 hidden units in each layer. 

The animation below shows generated digits (randomly sampled using a fixed seed) from the model during training for `K=0,1,...,75` epochs using the natural ordering of the inputs to the network:

<p align="center">
  <img src ="https://github.com/e-hulten/made/blob/master/gifs/model_500.gif">
</p>

The next figure shows reconstructed digits from the test set using the same one hidden layer network. The middle row shows the actual reconstructions, i.e., the outputs of the MADE with the images in the top row as inputs. The bottom row shows reconstructions in the sense that the pixels are sampled from the conditional Bernoulli distributions defined by the reconstructions in the middle row, hence the noisiness of the bottom row. 

<p align="center">
  <img src ="https://github.com/e-hulten/made/blob/master/reconstruct/reconstruct_75_epochs.png">
</p>


The above results come from a Bernoulli MADE as described in [1], but we have also added functionality to experiment with a MADE with Gaussian conditionals that is able to estimate real-valued densities. I will add plots for that later. 

[1] https://arxiv.org/pdf/1502.03509.pdf

## TODO:
* Add plots from Gaussian MADE. 
* Modify plotting functions to enable plotting models trained using a random ordering of the inputs.
* Add functionality to create ensembles at test time. 
