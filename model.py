import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class MaskedSum(nn.Linear):
    def __init__(self, n_in, n_out):
        super().__init__(n_in, n_out)

    def initialise_mask(self, mask):
        self.mask = mask

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)


class MaskedSumCompanion(nn.Linear):
    def __init__(self, n_in, n_out):
        super().__init__(n_in, n_out, bias=False)

    def initialise_mask(self, mask):
        self.mask = mask

    def forward(self, x):
        return F.linear(torch.ones_like(x), self.mask * self.weight, self.bias)


class MADE(nn.Module):
    def __init__(
        self, n_in, hidden_dims, random_order=False, seed=None, gaussian=False
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = 2 * n_in if gaussian is True else n_in
        self.hidden_dims = hidden_dims
        self.random_order = random_order
        self.seed = seed
        self.gaussian = gaussian
        self.relu = torch.nn.ReLU(inplace=True)
        self.layers = []

        # input -> first hidden
        self.layers.append(MaskedSum(n_in, self.hidden_dims[0]))
        self.layers.append(self.relu)
        # hidden -> hidden
        for l in range(1, len(hidden_dims)):
            self.layers.append(MaskedSum(hidden_dims[l - 1], hidden_dims[l]))
            self.layers.append(self.relu)
        # hidden -> output
        self.layers.append(MaskedSum(hidden_dims[-1], self.n_out))

        # create model
        self.model = nn.Sequential(*self.layers)
        # get masks for the masked activations
        self.create_masks()

    def forward(self, x):
        return self.model(x) if self.gaussian else torch.sigmoid(self.model(x))

    def create_masks(self):
        np.random.seed(self.seed)
        self.masks = {}
        L = len(self.hidden_dims)  # number of hidden layers
        D = self.n_in  # number of inputs

        # if false, use the natural ordering [1,2,...,D]
        self.masks[0] = (
            np.random.permutation(self.n_in)
            if self.random_order
            else np.arange(self.n_in)
        )

        # set the connectivity number m for the hidden layers
        # m ~ DiscreteUniform[min_{prev_layer}(m), D-1]
        for l in range(L):
            self.masks[l + 1] = np.random.randint(
                self.masks[l].min(), D - 1, size=self.hidden_dims[l]
            )

        self.mask_matrix = []
        # create mask matrix for input->hidden_1->...->hidden_L
        # (i.e., excluding hidden_L->output)
        for mask_num in range(len(self.masks) - 1):
            m = self.masks[mask_num]  # current layer
            m_next = self.masks[mask_num + 1]  # next layer
            M = torch.zeros(len(m_next), len(m))  # mask matrix
            for i in range(len(m_next)):
                M[i, :] = torch.from_numpy((m_next[i] >= m).astype(int))
            self.mask_matrix.append(M)

        # create mask matrix for hidden_L->output
        m = self.masks[L]
        m_out = self.masks[0]
        M_out = torch.zeros(len(m_out), len(m))
        for i in range(len(m_out)):
            M_out[i, :] = torch.from_numpy((m_out[i] > m).astype(int))

        M_out = torch.cat((M_out, M_out), dim=0) if self.gaussian is True else M_out

        self.mask_matrix.append(M_out)

        # get masks for the layers of self.model
        mask_iter = iter(self.mask_matrix)
        for module in self.model.modules():
            if isinstance(module, MaskedSum):
                module.initialise_mask(next(mask_iter))


class MADECompanion(nn.Module):
    def __init__(
        self, n_in, hidden_dims, random_order=False, seed=None, gaussian=False
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = 2 * n_in if gaussian is True else n_in
        self.hidden_dims = hidden_dims
        self.random_order = random_order
        self.seed = seed
        self.gaussian = gaussian
        self.layers = nn.ModuleList()
        self.companion = nn.ModuleList()

        # input -> first hidden
        self.layers.append(MaskedSum(n_in, self.hidden_dims[0]))
        self.companion.append(MaskedSumCompanion(n_in, self.hidden_dims[0]))
        # hidden -> hidden
        for l in range(1, len(hidden_dims)):
            self.layers.append(MaskedSum(hidden_dims[l - 1], hidden_dims[l]))
            self.companion.append(
                MaskedSumCompanion(hidden_dims[l - 1], hidden_dims[l])
            )
        # hidden -> output
        self.layers.append(MaskedSum(hidden_dims[-1], self.n_out))
        self.companion.append(MaskedSumCompanion(hidden_dims[-1], self.n_out))

        # get masks for the masked activations
        self.create_masks()

    def forward(self, x):
        self.set_masks()
        for (layer, companion) in zip(self.layers[:-1], self.companion[:-1]):
            x = F.relu(layer(x) + companion(x))

        x = self.layers[-1](x) + self.companion[-1](x)
        return torch.sigmoid(x) if not self.gaussian else x

    def create_masks(self):
        np.random.seed(self.seed)
        self.masks = {}
        L = len(self.hidden_dims)
        D = self.n_in

        self.masks[0] = (
            np.random.permutation(self.n_in)
            if self.random_order
            else np.arange(self.n_in)
        )

        # set the connectivity number m for the hidden layers
        # m ~ DiscreteUniform[min_{prev_layer}(m), D-1]
        for l in range(L):
            self.masks[l + 1] = np.random.randint(
                self.masks[l].min(), D - 1, size=self.hidden_dims[l]
            )

        # initialize array for mask matrices of different sizes
        self.mask_matrix = []
        # create mask matrix for input->hidden_1->...->hidden_L
        for mask_num in range(len(self.masks) - 1):
            m = self.masks[mask_num]  # current layer
            m_next = self.masks[mask_num + 1]  # next layer
            M = torch.zeros(len(m_next), len(m))  # mask matrix
            for i in range(len(m_next)):
                M[i, :] = torch.from_numpy((m_next[i] >= m).astype(int))
            self.mask_matrix.append(M)

        # create mask matrix for hidden_L->output
        m = self.masks[L]
        m_out = self.masks[0]
        M_out = torch.zeros(len(m_out), len(m))

        for i in range(len(m_out)):
            M_out[i, :] = torch.from_numpy((m_out[i] > m).astype(int))

        M_out = torch.cat((M_out, M_out), dim=0) if self.gaussian is True else M_out
        self.mask_matrix.append(M_out)

    def set_masks(self):
        # get masks for the layers of self.model
        mask_iter = iter(self.mask_matrix)
        for (layer, companion) in zip(self.layers, self.companion):
            mask = next(mask_iter)
            layer.initialise_mask(mask)
            companion.initialise_mask(mask)

