
import numpy as np
from math import sqrt

import torch
import torch.nn as nn


class DPLinear(nn.Module):
    def __init__(self, nested: nn.Linear):
        super().__init__()
        self.weight = nested.weight.detach()
        self.bias = nested.bias.detach()
        self.in_features = nested.in_features
        self.out_features = nested.out_features

    def forward(self, x):
        x.save()

        # append bias as last column
        init_slb = torch.cat([self.weight, self.bias.unsqueeze(1)], dim=1)

        x.lb = init_slb
        x.ub = init_slb
        x.slb = init_slb
        x.sub = init_slb
        # loop layer backwards
        for i in range(x.layers - 1, -1, -1):
            x.lb = x.resolve(x.lb, i, lower=True)
            x.ub = x.resolve(x.ub, i, lower=False)
        x.is_relu = False
        return x


class DPReLU(nn.Module):
    def __init__(self, in_features):
        super(DPReLU, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        # have lambdas as trainable parameter
        self.lam = torch.nn.Parameter(torch.ones(in_features))  # TODO: pick different lambdas here? train them?

    def forward(self, x):
        x.save()

        lb, ub = x.lb, x.ub

        # cases 1-3.2
        mask_1, mask_2 = lb.ge(0), ub.le(0)
        mask_3 = ~(mask_1 | mask_2)
        # this should be the right area minimizing heuristic (starting from `ones` lambdas)
        # self.lam.data = self.lam.where(ub.gt(-lb), torch.zeros_like(self.lam))
        a = torch.where((ub - lb) == 0, torch.ones_like(ub), ub / (ub - lb + 1e-6))
        
        x.lb = lb * mask_1 + self.lam * lb * mask_3
        x.ub = ub * mask_1 + ub * mask_3
        curr_slb = 1 * mask_1 + self.lam * mask_3
        curr_sub = 1 * mask_1 + a * mask_3
        bias = - lb * a * mask_3

        # only save slb and sub as vectors, which we know encodes a diag matrix
        x.slb = torch.cat([curr_slb.unsqueeze(0), torch.zeros(len(lb)).unsqueeze(0)], dim=0)
        x.sub = torch.cat([curr_sub.unsqueeze(0), bias.unsqueeze(0)], dim=0)
        x.is_relu = True
        return x


class DPNormalization(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.mean = 0.1307
        self.sigma = 0.3081
        self.in_features = in_features
        self.out_features = in_features

    def forward(self, x):
        x.save()
        x.lb = (x.lb - self.mean) / self.sigma
        x.ub = (x.ub - self.mean) / self.sigma
        ones = torch.ones(x.lb.shape)
        x.slb = torch.cat([(1/self.sigma) * torch.diag(ones), (-self.mean/self.sigma) * ones.unsqueeze(1)], dim=1)
        x.sub = x.slb
        x.is_relu = False
        return x


class Validator(nn.Module):
    def __init__(self, in_features, out_features=-1, bias=False, true_label=None, verbose=False):
        # ignore out_features (just kept them for inheritance with linear)
        super(Validator, self).__init__()
        self.in_features = in_features
        self.out_features = in_features - 1
        self.true_label = true_label
        self.verbose = verbose
        # weights can be seen as -1 * identity matrix, with
        # one column inserted that is all ones for the right label we compare to
        # (also add zero bias at the end)

        ids = -1 * torch.diag(torch.ones(self.in_features - 1))
        ones = torch.ones(self.in_features - 1).unsqueeze(1)
        zeros = torch.zeros_like(ones)
        self.weights = torch.cat((ids[:, :true_label], ones, ids[:, true_label:], zeros), dim=1)
    
    def forward(self, x):
        x.save()

        if self.true_label is None:
            print("True label must not be None !")

        curr_slb = self.weights
        # loop layer backwards
        for i in range(x.layers - 1, -1, -1):
            curr_slb = x.resolve(curr_slb, i, lower=True)

        if self.verbose:
            print("--- Validator\nFinal lower bound: ", curr_slb)
        return curr_slb, x.lb, x.ub


class DPPrintLayer(nn.Module):
    def __init__(self, in_features):
        super(DPPrintLayer, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

    def forward(self, x):
        print('--- DPPrintLayer')
        print('lower bounds: ', x.lb)
        print('upper bounds: ', x.ub)
        # print('slb:', x.slb)
        # print('sub:', x.sub)
        return x


class DPConv2D(nn.Module):
    def __init__(self, nested: nn.Conv2d, in_features: int):
        """
        Initializes a transformer module for 2D Conv layers.
        :param nested: The instantiated Conv layer from the loaded model.
        :param in_features: This should be an integer, denoting the total amount of input neurons: a tensor AxBxC would
                            translate into in_features=A*B*C
        """
        super().__init__()
        # first, the easy stuff, simply copy attributes:
        self.weight = nested.weight.detach()
        self.bias = nested.bias.detach()
        # padding, stride and kernel_size automatically get expanded into a tuple by PyTorch
        self.padding = nested.padding[0]
        self.stride = nested.stride[0]
        self.kernel_size = nested.kernel_size[0]
        # bit more complicated, have to calculate in and out dimensions
        self.in_features = in_features
        self.in_channels = nested.in_channels
        self.out_channels = nested.out_channels
        assert self.in_features % self.in_channels == 0, "FATAL: Input mismatch: in_channels !"
        # given the current state of build_verifier_network, we kinda need to assume quadratic tensors,
        # since we only have the in_features information, but we know nothing about spatial dimensions
        assert sqrt(self.in_features / self.in_channels).is_integer(), "FATAL: Input mismatch: spatial dimensions !"
        self.in_height = int(sqrt(self.in_features / self.in_channels))
        self.in_width = int(sqrt(self.in_features / self.in_channels))
        self.out_height = int((self.in_height + 2 * self.padding - self.kernel_size) / self.stride + 1)
        self.out_width = int((self.in_width + 2 * self.padding - self.kernel_size) / self.stride + 1)
        # again, have to assume that all tensors are quadratic in the spatial dimensions
        assert self.out_width == self.out_height, "FATAL: Output mismatch: spatial dimensions !"
        self.out_features = self.out_channels * self.out_height * self.out_width
        # build 2D matrix from kernel weights
        self.matrix = self.build_matrix()

    def build_matrix(self):
        # the basic idea is to first treat padding as normal input dimensions and cut it away later.
        # this greatly simplifies the filling process.
        padded_in_width = self.in_width + 2 * self.padding
        padded_in_height = self.in_height + 2 * self.padding
        padded_in_spatial_dim = padded_in_height * padded_in_width

        # prepare result matrix
        res = torch.zeros((self.out_height * self.out_width * self.out_channels,
                           padded_in_spatial_dim * self.in_channels))

        # build row fillers
        len_row_fillers = (self.in_channels - 1) * padded_in_spatial_dim +\
                          (self.kernel_size - 1) * padded_in_width + self.kernel_size
        row_fillers = torch.zeros((self.out_channels, len_row_fillers))
        for out_ch_idx in range(self.out_channels):
            for in_ch_idx in range(self.in_channels):
                for kernel_row_idx in range(self.kernel_size):
                    start_idx = in_ch_idx * padded_in_spatial_dim + kernel_row_idx * padded_in_width
                    row_fillers[out_ch_idx, start_idx: start_idx + self.kernel_size] =\
                        self.weight[out_ch_idx, in_ch_idx, kernel_row_idx]

        # fill result matrix with row fillers
        for out_ch_idx in range(self.out_channels):
            for out_height_idx in range(self.out_height):
                for out_width_idx in range(self.out_width):
                    row_offset = out_height_idx * self.stride * padded_in_width + out_width_idx * self.stride
                    out_neuron_idx = out_ch_idx * self.out_height * self.out_width +\
                                     out_height_idx * self.out_width + out_width_idx
                    res[out_neuron_idx, row_offset: row_offset + len_row_fillers] = row_fillers[out_ch_idx]

        # cut away padding:
        del_cols = []
        for in_ch_idx in range(self.in_channels):
            for in_height_idx in range(padded_in_height):
                for in_width_idx in range(padded_in_width):
                    if in_width_idx < self.padding or in_width_idx >= self.padding + self.in_width:
                        del_cols.append(
                            in_ch_idx * padded_in_spatial_dim + in_height_idx * padded_in_width + in_width_idx)
                if in_height_idx < self.padding or in_height_idx >= self.padding + self.in_height:
                    start_idx = in_ch_idx * padded_in_spatial_dim + in_height_idx * padded_in_width
                    del_cols = del_cols + list(range(start_idx, start_idx + padded_in_width))

        # remove duplicates:
        del_cols = list(np.unique(np.array(del_cols)))
        # couldn't find a way to delete cols from a tensor, thus:
        res = torch.from_numpy(np.delete(res.numpy(), del_cols, axis=1))

        # attach bias as last column:
        res = torch.cat([res, torch.repeat_interleave(self.bias, self.out_width * self.out_height).unsqueeze(1)], dim=1)

        return res

    def forward(self, x):
        x.save()

        if x.is_relu:
            assert x.slb.size()[1] == self.in_features, "FATAL: ReLU input has wrong number of neurons !"
        else:
            assert x.slb.size()[0] == self.in_features, "FATAL: Input has wrong number of neurons !"

        init_slb = self.matrix

        # copied from DPLinear
        x.lb = init_slb
        x.ub = init_slb
        x.slb = init_slb
        x.sub = init_slb
        # loop layer backwards
        for i in range(x.layers - 1, -1, -1):
            x.lb = x.resolve(x.lb, i, lower=True)
            x.ub = x.resolve(x.ub, i, lower=False)
        x.is_relu = False
        return x
