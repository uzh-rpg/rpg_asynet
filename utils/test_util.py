import torch
import numpy as np

try:
    import sparseconvnet as scn
except Exception:
    print("Could not import sparseconvnet")

import layers.site_enum as ascn1
import layers.conv_layer_2D as ascn2
import layers.max_pool as ascn_max


def createInput(nIn=1, spatial_dimensions=[10, 20], asynchronous_input=True, sequence_length=3,
                simplified=False):
    """Creates an input for asynchronous or batch sparse network"""
    # np.random.seed(3)

    if not asynchronous_input and sequence_length != 1:
        raise ValueError('Expected the sequence length to be 1 for batch input Got sequence length %s' %
                         sequence_length)
    asyn_input = []
    asyn_update_locations = []
    batch_input = np.zeros(spatial_dimensions + [nIn])
    spatial_volume = np.prod(spatial_dimensions)
    for i in range(sequence_length):
        if simplified:
            nr_cell_updates = 1
            random_features = np.random.randint(-1, 1, [nr_cell_updates, nIn])
        else:
            high_threshold = min(spatial_volume / 2, 200)
            nr_cell_updates = np.random.randint(low=1, high=max(high_threshold, 2))
            random_features = np.random.randint(-10, 10, [nr_cell_updates, nIn])

        # Exclude feature updates for difference of zero
        random_features[np.sum(random_features ** 2, axis=-1) == 0, :] = np.ones([1, nIn])
        random_permutation = np.random.permutation(spatial_volume)[:nr_cell_updates].astype(np.int)
        asyn_input_i = np.zeros([spatial_volume, nIn])
        asyn_input_i[random_permutation, :] = random_features
        asyn_input_i = asyn_input_i.reshape(spatial_dimensions + [nIn])

        batch_input = batch_input + asyn_input_i
        asyn_input.append(batch_input.copy())
        asyn_locations_i = np.stack(np.unravel_index(random_permutation, dims=spatial_dimensions), axis=-1)
        asyn_update_locations.append(asyn_locations_i.copy())

    batch_update_locations = np.stack(((batch_input ** 2).sum(axis=-1) != 0).nonzero(), axis=-1)

    if asynchronous_input:
        return batch_input, batch_update_locations, asyn_input, asyn_update_locations
    else:
        return batch_input, batch_update_locations


def createConvBatchNormLayers(nLayers, nChannels, use_bias, use_batch_norm, dimension=2, facebook_layer=True, kernel_size=3):
    """Creates sparse layers"""
    kernels = []
    if dimension == 1:
        asyn_conv_layer = ascn1.asynSparseConvolution1D
    elif dimension == 2:
        asyn_conv_layer = ascn2.asynSparseConvolution2D

    if use_bias:
        bias = []
    if use_batch_norm:
        batch_norm_param = []
    for i_layer in range(1, nLayers + 1):
        kernels.append(np.random.uniform(-10.0, 10.0, [kernel_size**dimension, 1, nChannels[i_layer - 1],
                                                       nChannels[i_layer]]))
        if use_bias:
            bias.append(np.random.uniform(-10.0, 10.0, [nChannels[i_layer]]))
        if use_batch_norm:
            learned_scale = np.random.uniform(-1.0, 1.0, [nChannels[i_layer]])
            learned_shift = np.random.uniform(-1.0, 1.0, [nChannels[i_layer]])
            running_mean = np.random.uniform(-1.0, 1.0, [nChannels[i_layer]])
            running_var = np.random.uniform(0.1, 1.0, [nChannels[i_layer]])
            batch_norm_param.append([learned_scale, learned_shift, running_mean, running_var])

    asyn_conv_layers = []
    asyn_bn_layers = []
    for i_layer in range(1, nLayers + 1):
        asyn_conv_layers.append(asyn_conv_layer(dimension=dimension, nIn=nChannels[i_layer - 1],
                                                nOut=nChannels[i_layer],
                                                filter_size=kernel_size,
                                                first_layer=(i_layer == 1),
                                                use_bias=use_bias))
        asyn_conv_layers[i_layer - 1].weight.data = torch.squeeze(torch.tensor(kernels[i_layer - 1],
                                                                               dtype=torch.float32), dim=1)
        if use_bias:
            asyn_conv_layers[i_layer - 1].bias.data = torch.tensor(bias[i_layer - 1], dtype=torch.float32)
        if use_batch_norm:
            asyn_bn_layers.append(torch.nn.BatchNorm1d(nChannels[i_layer - 1], eps=1e-4, momentum=0.9))
            asyn_bn_layers[i_layer - 1].weight.data = torch.tensor(batch_norm_param[i_layer - 1][0],
                                                                   dtype=torch.float64)
            asyn_bn_layers[i_layer - 1].bias.data = torch.tensor(batch_norm_param[i_layer - 1][1],
                                                                 dtype=torch.float64)
            asyn_bn_layers[i_layer - 1].running_mean.data = torch.tensor(batch_norm_param[i_layer - 1][2],
                                                                         dtype=torch.float64)
            asyn_bn_layers[i_layer - 1].running_var.data = torch.tensor(batch_norm_param[i_layer - 1][3],
                                                                        dtype=torch.float64)
            asyn_bn_layers[i_layer - 1].eval()

    if facebook_layer:
        sparse_conv_layers = []
        sparse_bn_layers = []
        for i_layer in range(1, nLayers + 1):
            sparse_conv_layers.append(scn.SubmanifoldConvolution(dimension=dimension, nIn=nChannels[i_layer - 1],
                                                                 nOut=nChannels[i_layer],
                                                                 filter_size=kernel_size,
                                                                 bias=use_bias))
            sparse_conv_layers[i_layer - 1].weight.data = torch.tensor(kernels[i_layer - 1],
                                                                       dtype=torch.float32)
            if use_bias:
                sparse_conv_layers[i_layer - 1].bias.data = torch.tensor(bias[i_layer - 1],
                                                                         dtype=torch.float32)
            if use_batch_norm:
                sparse_bn_layers.append(scn.BatchNormalization(nChannels[i_layer]))
                sparse_bn_layers[i_layer - 1].weight.data = torch.tensor(batch_norm_param[i_layer - 1][0],
                                                                         dtype=torch.float32)
                sparse_bn_layers[i_layer - 1].bias.data = torch.tensor(batch_norm_param[i_layer - 1][1],
                                                                       dtype=torch.float32)
                sparse_bn_layers[i_layer - 1].running_mean.data = torch.tensor(batch_norm_param[i_layer - 1][2],
                                                                               dtype=torch.float32)
                sparse_bn_layers[i_layer - 1].running_var.data = torch.tensor(batch_norm_param[i_layer - 1][3],
                                                                              dtype=torch.float32)
                sparse_bn_layers[i_layer - 1].eval()

        return sparse_conv_layers, sparse_bn_layers, asyn_conv_layers, asyn_bn_layers
    else:
        batch_asyn_conv_layers = []
        batch_asyn_bn_layers = []
        for i_layer in range(1, nLayers+1):
            batch_asyn_conv_layers.append(asyn_conv_layer(dimension=dimension, nIn=nChannels[i_layer-1],
                                                          nOut=nChannels[i_layer],
                                                          filter_size=kernel_size,
                                                          first_layer=(i_layer == 1),
                                                          use_bias=use_bias),)
            batch_asyn_conv_layers[i_layer-1].weight.data = torch.squeeze(torch.tensor(kernels[i_layer-1],
                                                                          dtype=torch.float32), dim=1)

            if use_bias:
                batch_asyn_conv_layers[i_layer - 1].bias.data = torch.tensor(bias[i_layer - 1], dtype=torch.float32)

            if use_batch_norm:
                batch_asyn_bn_layers.append(torch.nn.BatchNorm1d(nChannels[i_layer - 1], eps=1e-4, momentum=0.9))
                batch_asyn_bn_layers[i_layer - 1].weight.data = torch.tensor(batch_norm_param[i_layer - 1][0],
                                                                             dtype=torch.float64)
                batch_asyn_bn_layers[i_layer - 1].bias.data = torch.tensor(batch_norm_param[i_layer - 1][1],
                                                                           dtype=torch.float64)
                batch_asyn_bn_layers[i_layer - 1].running_mean.data = torch.tensor(batch_norm_param[i_layer - 1][2],
                                                                                   dtype=torch.float64)
                batch_asyn_bn_layers[i_layer - 1].running_var.data = torch.tensor(batch_norm_param[i_layer - 1][3],
                                                                                  dtype=torch.float64)
                batch_asyn_bn_layers[i_layer - 1].eval()

        return batch_asyn_conv_layers, batch_asyn_bn_layers, asyn_conv_layers, asyn_bn_layers


def createConvCPPmLayers(nLayers, nChannels, use_bias, facebook_layer=True, kernel_size=3):
    """Creates sparse layers"""
    import layers.conv_layer_2D_cpp as ascn_cpp

    dimension = 2
    kernels = []
    asyn_conv_layer = ascn_cpp.asynSparseConvolution2Dcpp

    if use_bias:
        bias = []

    for i_layer in range(1, nLayers + 1):
        kernels.append(np.random.uniform(-10.0, 10.0, [kernel_size**dimension, 1, nChannels[i_layer - 1],
                                                       nChannels[i_layer]]))
        if use_bias:
            bias.append(np.random.uniform(-10.0, 10.0, [nChannels[i_layer]]))

    asyn_conv_layers = []

    for i_layer in range(1, nLayers + 1):
        asyn_conv_layers.append(asyn_conv_layer(dimension=dimension, nIn=nChannels[i_layer - 1],
                                                nOut=nChannels[i_layer],
                                                filter_size=kernel_size,
                                                first_layer=(i_layer == 1),
                                                use_bias=use_bias,
                                                debug=False))
        asyn_conv_layers[i_layer - 1].setParameters(kernels[i_layer - 1], bias[i_layer - 1])

    if facebook_layer:
        sparse_conv_layers = []
        for i_layer in range(1, nLayers + 1):
            sparse_conv_layers.append(scn.SubmanifoldConvolution(dimension=dimension, nIn=nChannels[i_layer - 1],
                                                                 nOut=nChannels[i_layer],
                                                                 filter_size=kernel_size,
                                                                 bias=use_bias))
            sparse_conv_layers[i_layer - 1].weight.data = torch.tensor(kernels[i_layer - 1],
                                                                       dtype=torch.float32)
            if use_bias:
                sparse_conv_layers[i_layer - 1].bias.data = torch.tensor(bias[i_layer - 1],
                                                                         dtype=torch.float32)

        return sparse_conv_layers, asyn_conv_layers
    else:
        batch_asyn_conv_layers = []
        for i_layer in range(1, nLayers+1):
            batch_asyn_conv_layers.append(asyn_conv_layer(dimension=dimension, nIn=nChannels[i_layer-1],
                                                          nOut=nChannels[i_layer],
                                                          filter_size=kernel_size,
                                                          first_layer=(i_layer == 1),
                                                          use_bias=use_bias,
                                                          debug=False),)
            batch_asyn_conv_layers[i_layer-1].setParameters(kernels[i_layer - 1], bias[i_layer - 1])

        return batch_asyn_conv_layers, asyn_conv_layers


def createMaxLayers(dimension=2, facebook_layer=True, pool_size=3, pool_stride=1, padding_mode='valid'):
    """Creates max pooling layers"""
    asyn_max_layer = ascn_max.asynMaxPool(dimension=dimension, filter_size=pool_size, filter_stride=pool_stride,
                                          padding_mode=padding_mode)

    if facebook_layer:
        if padding_mode != 'valid':
            raise ValueError('Expected padding mode valid for facebook implementation. Got mode: %s' % padding_mode)
        sparse_max_layer = scn.MaxPooling(dimension, pool_size, pool_stride)

        return sparse_max_layer, asyn_max_layer
    else:
        batch_asyn_max_layer = ascn_max.asynMaxPool(dimension=dimension, filter_size=pool_size,
                                                    filter_stride=pool_stride, padding_mode=padding_mode)
        return batch_asyn_max_layer, asyn_max_layer