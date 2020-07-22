#! /home/dani/anaconda3/envs/sparse_conv/bin/python3.7
"""
Example command: python -m unittests.sparse_conv2D_cpp_test
"""
import numpy as np
import torch
import tqdm
import unittest

import layers.conv_layer_2D as ascn
import layers.conv_layer_2D_cpp as ascn_cpp
import sparseconvnet as scn
import utils.test_util as test_util


np.random.seed(3)


class TestConv2DSync(unittest.TestCase):
    def test_fb_sparse_convolution1d_comparison(self):
        """Tests if the output of one layer asynchronous CNN in one time step is equal to the python implementation"""
        debug = False
        # Create Input
        nIn = 5
        nOut = 8
        use_bias = True
        dimension = 2
        filter_size = 3
        spatial_dimensions = [250, 250]

        kernel = np.random.uniform(-10.0, 10.0, [filter_size ** dimension, 1, nIn, nOut])
        if use_bias:
            bias = np.random.uniform(-10.0, 10.0, [nOut])
        out = test_util.createInput(nIn=nIn, spatial_dimensions=spatial_dimensions, asynchronous_input=False,
                                    sequence_length=1, simplified=False)
        batch_input, batch_update_locations = out

        # Asynchronous Sparse Convolution
        asyn_conv_layer_1 = ascn.asynSparseConvolution2D(dimension=dimension, nIn=nIn, nOut=nOut,
                                                         filter_size=filter_size, first_layer=True, use_bias=use_bias)
        asyn_conv_layer_1.weight.data = torch.squeeze(torch.tensor(kernel, dtype=torch.float32), dim=1)
        if use_bias:
            asyn_conv_layer_1.bias.data = torch.tensor(bias, dtype=torch.float32)

        # t1 = perf_counter()

        asyn_output = asyn_conv_layer_1.forward(update_location=torch.tensor(batch_update_locations),
                                                feature_map=torch.tensor(batch_input),
                                                active_sites_map=None,
                                                rule_book_input=None,
                                                rule_book_output=None)

        # dt_py = perf_counter() - t1
        # print("Python implementation: ", dt_py)

        # Asynchronous Sparse Convolution cpp
        conv_cpp = ascn_cpp.asynSparseConvolution2Dcpp(dimension, nIn, nOut, filter_size, True, use_bias, debug)
        conv_cpp.setParameters(kernel, bias)

        out = conv_cpp.forward(batch_update_locations, feature_map=batch_input)

        np.testing.assert_almost_equal(out[1], asyn_output[1].float().data.cpu().numpy(), decimal=4)
        np.testing.assert_almost_equal(out[0], asyn_output[0].float().data.cpu().numpy(), decimal=4)
        np.testing.assert_almost_equal(out[2], asyn_output[2].float().data.cpu().numpy(), decimal=4)

    def test_sparse_convolution_multiple_layers(self):
        """Tests if multiple layers of asynchronous CNN result in the same output as the facebook implementation"""
        print('Batch N-Layer Test')
        for i_test in tqdm.tqdm(range(100)):
            # print('Test: %s' % i_test)
            # print('#######################')
            # print('#       New Test      #')
            # print('#######################')
            use_relu = False
            use_bias = True
            nLayers = 4  # 4
            dimension = 2
            nChannels = [1, 5, 7, 5, 10]
            spatial_dimensions = [250, 250]
            out = test_util.createInput(nIn=nChannels[0], spatial_dimensions=spatial_dimensions, asynchronous_input=False,
                                        sequence_length=1, simplified=False)
            batch_input, batch_update_locations = out

            out = test_util.createConvCPPmLayers(nLayers, nChannels, use_bias, facebook_layer=True)
            sparse_conv_layers, asyn_conv_layers = out

            # Sparse convolution layer implementation by facebook
            spatial_size = torch.LongTensor(batch_input.shape[:dimension])
            select_indices = tuple(batch_update_locations.T)
            features = batch_input[select_indices]
            input_layer = scn.InputLayer(dimension, spatial_size, mode=3)
            output_layer = scn.SparseToDense(dimension=dimension, nPlanes=nChannels[-1])

            fb_output = input_layer([torch.LongTensor(batch_update_locations), torch.FloatTensor(features)])
            for i_layer in range(0, nLayers):
                fb_output = sparse_conv_layers[i_layer](fb_output)

            fb_output = output_layer(fb_output)

            # Layer 1
            x_cpp = asyn_conv_layers[0].forward(update_location=torch.tensor(batch_update_locations),
                                                feature_map=torch.tensor(batch_input))

            for i_layer in range(1, nLayers):
                if use_relu:
                    feature_input = np.max(x_cpp[1], 0)
                else:
                    feature_input = x_cpp[1]

                x_cpp = asyn_conv_layers[i_layer].forward(update_location=x_cpp[0], feature_map=feature_input,
                                                          active_sites_map=x_cpp[2], rule_book=x_cpp[3])

            np.testing.assert_almost_equal(torch.squeeze(fb_output, dim=0).data.cpu().numpy().transpose([1, 2, 0]),
                                           x_cpp[1], decimal=0)


    def test_sparse_convolution1d_multiple_layers_asynchronous(self):
        """Tests if multiple layers of asynchronous CNN result in the same output as the facebook implementation"""
        print('Asynchronous N-Layer Test')
        for i_test in tqdm.tqdm(range(100)):
            # print('Test: %s' % i_test)
            # print('#######################')
            # print('#       New Test      #')
            # print('#######################')
            use_relu = False
            use_bias = True
            nLayers = 4
            nChannels = [1, 5, 7, 5, 10]
            sequence_length = 10
            spatial_dimensions = [250, 300]
            out = test_util.createInput(nIn=nChannels[0], spatial_dimensions=spatial_dimensions,
                                        asynchronous_input=True, sequence_length=sequence_length, simplified=False)
            batch_input, batch_update_locations, asyn_input, asyn_update_locations = out

            out = test_util.createConvCPPmLayers(nLayers, nChannels, use_bias, facebook_layer=False)
            batch_asyn_conv_layers, asyn_conv_layers = out

            # ---- Batch Asynchronous Sparse Convolution ----
            batch_locations = batch_update_locations
            batch_dense_input = batch_input

            # Layer 1
            x_batch = batch_asyn_conv_layers[0].forward(update_location=batch_locations,
                                                        feature_map=batch_dense_input,
                                                        active_sites_map=None, rule_book=None)

            for i_layer in range(1, nLayers):
                if use_relu:
                    feature_input = np.maximum(x_batch[1], 0)
                else:
                    feature_input = x_batch[1]

                x_batch = batch_asyn_conv_layers[i_layer].forward(update_location=x_batch[0], feature_map=feature_input,
                                                                  active_sites_map=x_batch[2],
                                                                  rule_book=x_batch[3])

            # ---- Asynchronous Sparse Convolution ----
            for i_sequence in range(sequence_length):
                # print('Time in Sequence: %s ' % i_sequence)
                # Layer 1 Input
                asyn_locations = torch.tensor(asyn_update_locations[i_sequence], dtype=torch.long)
                asyn_dense_input = torch.tensor(asyn_input[i_sequence], dtype=torch.float)

                x_asyn = asyn_conv_layers[0].forward(update_location=asyn_locations.long(),
                                                     feature_map=asyn_dense_input.float(),
                                                     active_sites_map=None,
                                                     rule_book=None)

                # x_asyn: new_update_events, output_feature_map, active_sites_map, rule_book_input, rule_book_output
                for i_layer in range(1, nLayers):
                    if use_relu:
                        feature_input = np.maximum(x_asyn[1], 0)
                    else:
                        feature_input = x_asyn[1]
                    x_asyn = asyn_conv_layers[i_layer].forward(update_location=x_asyn[0],
                                                               feature_map=feature_input,
                                                               active_sites_map=x_asyn[2],
                                                               rule_book=x_asyn[3])

            np.testing.assert_almost_equal(x_batch[1], x_asyn[1], decimal=0)


if __name__ == '__main__':
    unittest.main()
