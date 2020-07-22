import numpy as np


def computeFLOPS(active_sites, layer_list, sparse_model=False, layerwise=False):
    """Compute the number floating point operation for the network specified by the layer list"""
    number_flops = np.zeros([len(layer_list)])
    kernel_size = 3
    for j, i_layer in enumerate(layer_list):
        layer_name = i_layer[0]
        if layer_name == 'C':
            if sparse_model:
                number_flops[j] = active_sites[j] * (2 * i_layer[1]) * i_layer[2] + active_sites[j] * i_layer[1]
            else:
                number_flops[j] = active_sites[j + 1] * (2 * kernel_size * kernel_size * i_layer[1] - 1) * i_layer[2]
        elif layer_name == 'ClassicC':
            number_flops[j] = active_sites[j+1] * (2 * kernel_size * kernel_size * i_layer[1] - 1) * i_layer[2]
        elif layer_name == 'BNRelu' or layer_name == 'ClassicBNRelu':
            # BatchNorm can be merged with previous convolution layer
            # ReLu:
            number_flops[j] = active_sites[j] * layer_list[j - 1][2]
        elif layer_name == 'ClassicFC':
            number_flops[j] = 2 * i_layer[1] * i_layer[2]
        elif layer_name == 'MP':
            if layer_list[j + 2][0] != 'BNRelu':
                raise NotImplementedError('FLOP calculation for MaxPooling not followed by ConvLayer and BN is not'
                                          'implemented')
            number_flops[j] = kernel_size * kernel_size * active_sites[j + 2] * layer_list[j + 1][1]

    if layerwise:
        return number_flops
    else:
        return number_flops.sum()
