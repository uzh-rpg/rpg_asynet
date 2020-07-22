"""
Example command: python -m unittests.sparse_conv2D_test
"""
import numpy as np
import sparseconvnet as scn
import torch
import torch.nn.functional as F
import tqdm
import unittest

import layers.conv_layer_2D as ascn
from layers.site_enum import Sites
import utils.test_util as test_util


class TestConv2DSync(unittest.TestCase):
    def test_fb_sparse_convolution1d(self):
        dense_input = np.array([[[0, 0, 0, 1],
                                 [0, 0, 2, 4],
                                 [1, 0, 0, 0]]])
        kernel = np.array([[[0, -1,  2],
                            [1,  1, -1],
                            [0,  0,  0]]])
        locations_x, locations_y = np.squeeze(dense_input).nonzero()
        features = dense_input[0, locations_x, locations_y]
        spatial_size = torch.LongTensor([3, 4])
        input_layer = scn.InputLayer(2, spatial_size, mode=3)
        sparse_conv_layer = scn.SubmanifoldConvolution(dimension=2, nIn=1, nOut=1, filter_size=3, bias=False)
        sparse_conv_layer.weight.data = torch.squeeze(torch.tensor(kernel.flatten(),
                                                                   dtype=torch.float32))[:, None, None, None]
        output_layer = scn.SparseToDense(dimension=2, nPlanes=1)

        x = input_layer([torch.LongTensor(np.concatenate((locations_x[:, None], locations_y[:, None]), axis=-1)),
                         torch.FloatTensor(features)[:, None]])
        x = sparse_conv_layer(x)
        x = output_layer(x)

        dense_output = np.array([[[0,  0, 0, 1],
                                  [0,  0, 0, 5],
                                  [1,  0, 0, 0]]])

        self.assertListEqual(dense_output.flatten().squeeze().tolist(), torch.squeeze(x.flatten()).data.cpu().tolist())

    def test_fb_sparse_convolution1d_comparison(self):
        """Tests if the output of one layer asynchronous CNN in one time step is equal to the facebook implementation"""
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

        # Facebook Implementation
        spatial_size = torch.LongTensor(batch_input.shape[:dimension])
        input_layer = scn.InputLayer(dimension, spatial_size, mode=3)
        sparse_conv_layer = scn.SubmanifoldConvolution(dimension=dimension, nIn=nIn, nOut=nOut,
                                                       filter_size=filter_size, bias=use_bias)
        sparse_conv_layer.weight.data = torch.tensor(kernel, dtype=torch.float32)

        if use_bias:
            sparse_conv_layer.bias.data = torch.tensor(bias, dtype=torch.float32)
        output_layer = scn.SparseToDense(dimension=dimension, nPlanes=nOut)

        select_indices = tuple(batch_update_locations.T)
        features = batch_input[select_indices]

        fb_output = input_layer([torch.LongTensor(batch_update_locations), torch.FloatTensor(features)])
        fb_output = sparse_conv_layer(fb_output)
        fb_output = output_layer(fb_output)

        # Asynchronous Sparse Convolution
        asyn_conv_layer_1 = ascn.asynSparseConvolution2D(dimension=dimension, nIn=nIn, nOut=nOut,
                                                         filter_size=filter_size, first_layer=True, use_bias=use_bias)
        asyn_conv_layer_1.weight.data = torch.squeeze(torch.tensor(kernel, dtype=torch.float32), dim=1)
        if use_bias:
            asyn_conv_layer_1.bias.data = torch.tensor(bias, dtype=torch.float32)

        asyn_output = asyn_conv_layer_1.forward(update_location=torch.tensor(batch_update_locations),
                                                feature_map=torch.tensor(batch_input),
                                                active_sites_map=None,
                                                rule_book_input=None,
                                                rule_book_output=None)

        np.testing.assert_almost_equal(torch.squeeze(fb_output, dim=0).data.cpu().numpy().transpose([1, 2, 0]),
                                       asyn_output[1].float().data.cpu().numpy(), decimal=4)

    def test_sparse_convolution1d_asynchronous(self):
        """Test for 1D convolution with one layer in a asynchronous update"""
        print('Asynchronous 1-Layer Test')
        for i_test in tqdm.tqdm(range(10)):
            # print('Test: %s' % i_test)
            # print('#######################')
            # print('#       New Test      #')
            # print('#######################')
            nIn = 4
            nOut = 20
            sequence_length = 13
            use_bias = True
            use_batch_norm = True
            spatial_dimensions = [260, 250]

            out = test_util.createInput(nIn=nIn, spatial_dimensions=spatial_dimensions, asynchronous_input=True,
                                        sequence_length=sequence_length, simplified=False)
            batch_input, batch_update_locations, asyn_input, asyn_update_locations = out

            out = test_util.createConvBatchNormLayers(1, [nIn, nOut], use_bias, use_batch_norm, dimension=2,
                                                      facebook_layer=False)
            batch_conv_layers, batch_conv_bn_layers, asyn_conv_layers, asyn_bn_layers = out
            batch_conv_layer = batch_conv_layers[0]
            asyn_conv_layer = asyn_conv_layers[0]

            # Batch Computation
            batch_input = torch.tensor(batch_input)
            locations_batch = torch.tensor(batch_update_locations)
            batch_output = batch_conv_layer.forward(update_location=locations_batch.long(),
                                                    feature_map=batch_input.float(),
                                                    active_sites_map=None, rule_book_input=None, rule_book_output=None)
            if use_batch_norm:
                self.applyBatchNorm(batch_conv_bn_layers[0], batch_output[2].clone(), batch_output[1])

            # Asynchronous Update
            for time_i in range(len(asyn_input)):
                asyn_input_i = torch.tensor(asyn_input[time_i], dtype=torch.float32)
                asyn_locations_i = torch.tensor(asyn_update_locations[time_i], dtype=torch.float32)

                asyn_output = asyn_conv_layer.forward(update_location=asyn_locations_i.long(),
                                                      feature_map=asyn_input_i.float(),
                                                      active_sites_map=None, rule_book_input=None,
                                                      rule_book_output=None)

                if use_batch_norm:
                    self.applyBatchNorm(asyn_bn_layers[0], asyn_output[2].clone(), asyn_output[1])

            self.assertListEqual(torch.squeeze(batch_output[1]).data.cpu().tolist(),
                                 torch.squeeze(asyn_output[1]).data.cpu().tolist())

    def test_sparse_convolution1d_multiple_layers(self):
        """Tests if multiple layers of asynchronous CNN result in the same output as the facebook implementation"""
        print('Batch N-Layer Test')
        for i_test in tqdm.tqdm(range(10)):
            dimension = 2
            nLayers = 3
            nChannels = [3, 5, 10, 10]
            spatial_dimensions = [260, 250]
            use_bias = True
            use_batch_norm = True

            out = test_util.createInput(nIn=nChannels[0], spatial_dimensions=spatial_dimensions,
                                        asynchronous_input=False, sequence_length=1, simplified=False)
            batch_input, batch_update_locations = out

            out = test_util.createConvBatchNormLayers(nLayers, nChannels, use_bias, use_batch_norm, dimension=2,
                                                      facebook_layer=True)
            sparse_conv_layers, sparse_bn_layers, asyn_conv_layers, asyn_bn_layers = out

            # Sparse convolution layer implementation by facebook
            spatial_size = torch.LongTensor(batch_input.shape[:dimension])
            select_indices = tuple(batch_update_locations.T)
            features = batch_input[select_indices]
            input_layer = scn.InputLayer(dimension, spatial_size, mode=3)
            output_layer = scn.SparseToDense(dimension=dimension, nPlanes=nChannels[-1])

            fb_output = input_layer([torch.LongTensor(batch_update_locations), torch.FloatTensor(features)])
            for i_layer in range(0, nLayers):
                fb_output = sparse_conv_layers[i_layer](fb_output)
                if use_batch_norm:
                    fb_output = sparse_bn_layers[i_layer](fb_output)

            fb_output = output_layer(fb_output)

            # Batch Asynchronous Sparse Convolution
            locations_1 = torch.tensor(batch_update_locations, dtype=torch.long)
            dense_input = torch.tensor(batch_input, dtype=torch.float)

            # Layer 1
            x = asyn_conv_layers[0].forward(update_location=locations_1.long(),
                                            feature_map=dense_input.float(),
                                            active_sites_map=None, rule_book_input=None, rule_book_output=None)
            if use_batch_norm:
                self.applyBatchNorm(asyn_bn_layers[0], x[2].clone(), x[1])

            for i_layer in range(1, nLayers):
                x = asyn_conv_layers[i_layer].forward(update_location=x[0], feature_map=x[1], active_sites_map=x[2],
                                                      rule_book_input=x[3], rule_book_output=x[4])

                if use_batch_norm:
                    self.applyBatchNorm(asyn_bn_layers[i_layer], x[2].clone(), x[1])

            np.testing.assert_almost_equal(torch.squeeze(fb_output, dim=0).data.cpu().numpy().transpose([1, 2, 0]),
                                           x[1].float().data.cpu().numpy(), decimal=1)

    def test_sparse_convolution1d_multiple_layers_asynchronous(self):
        """Tests if multiple layers of asynchronous CNN result in the same output as the facebook implementation"""
        print('Asynchronous N-Layer Test')
        for i_test in tqdm.tqdm(range(10)):
            # print('Test: %s' % i_test)
            # print('#######################')
            # print('#       New Test      #')
            # print('#######################')
            use_relu = True
            use_bias = True
            use_batch_norm = True
            nLayers = 4
            nChannels = [1, 5, 7, 5, 10]
            sequence_length = 10
            spatial_dimensions = [260, 250]
            out = test_util.createInput(nIn=nChannels[0], spatial_dimensions=spatial_dimensions,
                                        asynchronous_input=True, sequence_length=sequence_length, simplified=False)
            batch_input, batch_update_locations, asyn_input, asyn_update_locations = out

            out = test_util.createConvBatchNormLayers(nLayers, nChannels, use_bias, use_batch_norm, dimension=2,
                                                      facebook_layer=False)
            batch_asyn_conv_layers, batch_asyn_bn_layers, asyn_conv_layers, asyn_bn_layers = out

            # ---- Batch Asynchronous Sparse Convolution ----
            batch_locations = torch.tensor(batch_update_locations, dtype=torch.long)
            batch_dense_input = torch.tensor(batch_input, dtype=torch.float)

            # Layer 1
            x_batch = batch_asyn_conv_layers[0].forward(update_location=batch_locations.long(),
                                                        feature_map=batch_dense_input.float(),
                                                        active_sites_map=None, rule_book_input=None,
                                                        rule_book_output=None)

            for i_layer in range(1, nLayers):
                if use_relu:
                    feature_input = F.relu(x_batch[1])
                else:
                    feature_input = x_batch[1]
                if use_batch_norm:
                    self.applyBatchNorm(batch_asyn_bn_layers[i_layer-1], x_batch[2].clone(), x_batch[1])

                x_batch = batch_asyn_conv_layers[i_layer].forward(update_location=x_batch[0], feature_map=feature_input,
                                                                  active_sites_map=x_batch[2],
                                                                  rule_book_input=x_batch[3],
                                                                  rule_book_output=x_batch[4])

            if use_batch_norm:
                self.applyBatchNorm(batch_asyn_bn_layers[-1], x_batch[2].clone(), x_batch[1])

            # ---- Asynchronous Sparse Convolution ----

            for i_sequence in range(sequence_length):
                # Layer 1 Input
                asyn_locations = torch.tensor(asyn_update_locations[i_sequence], dtype=torch.long)
                asyn_dense_input = torch.tensor(asyn_input[i_sequence], dtype=torch.float)

                x_asyn = asyn_conv_layers[0].forward(update_location=asyn_locations.long(),
                                                     feature_map=asyn_dense_input.float(),
                                                     active_sites_map=None,
                                                     rule_book_input=None,
                                                     rule_book_output=None)

                # x_asyn: new_update_events, output_feature_map, active_sites_map, rule_book_input, rule_book_output
                for i_layer in range(1, nLayers):
                    if use_relu:
                        feature_input = F.relu(x_asyn[1])
                    else:
                        feature_input = x_asyn[1]

                    if use_batch_norm:
                        self.applyBatchNorm(asyn_bn_layers[i_layer-1], x_asyn[2].clone(), x_asyn[1])

                    x_asyn = asyn_conv_layers[i_layer].forward(update_location=x_asyn[0],
                                                               feature_map=feature_input,
                                                               active_sites_map=x_asyn[2],
                                                               rule_book_input=x_asyn[3],
                                                               rule_book_output=x_asyn[4])

            if use_batch_norm:
                self.applyBatchNorm(asyn_bn_layers[-1], x_asyn[2].clone(), x_asyn[1])

            # self.assertListEqual(torch.squeeze(x_batch[1]).data.cpu().tolist(),
            #                      torch.squeeze(x_asyn[1]).data.cpu().tolist())
            np.testing.assert_almost_equal(x_batch[1].float().data.cpu().numpy(),
                                           x_asyn[1].float().data.cpu().numpy(), decimal=6)


    def applyBatchNorm(self, layer, bn_active_sites, feature_map_to_update):
        """
        Applies the batch norm layer to the the sparse features.

        :param layer: torch.nn.BatchNorm1d layer
        :param bn_active_sites: location of the active sites
        :param feature_map_to_update: feature map, result is stored in place to the tensor
        """
        bn_active_sites[bn_active_sites == Sites.NEW_INACTIVE_SITE.value] = 0
        active_sites = torch.squeeze(bn_active_sites.nonzero(), dim=-1)

        bn_input = torch.squeeze(feature_map_to_update[active_sites.split(1, dim=-1)], dim=1).T[None, :, :]
        sparse_bn_features = layer(bn_input)
        sparse_bn_features = torch.unsqueeze(torch.squeeze(sparse_bn_features, dim=0).T, dim=1)
        feature_map_to_update[active_sites.split(1, dim=-1)] = sparse_bn_features


if __name__ == '__main__':
    unittest.main()
