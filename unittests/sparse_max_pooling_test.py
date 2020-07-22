"""
Example command: python -m unittests.sparse_max_pooling_test
"""
import numpy as np
import sparseconvnet as scn
import torch
import tqdm
import unittest

import utils.test_util as test_util
import layers.max_pool as ascn


class TestMaxPoolSync(unittest.TestCase):
    def test_fb_maxpool(self):
        dimension = 2
        pool_size = 3
        pool_stride = 2

        batch_input = np.asarray([[0, 0, 0, 0, 0],
                                  [0, 1, 0, 2, 0],
                                  [0, 0, 0, 2, 0],
                                  [0, 0, 0, 2, 8],
                                  [0, 0, 0, 0, 0]]).astype(float)
        indices = batch_input.copy().nonzero()
        locations = np.concatenate([indices[0][:, None], indices[1][:, None]], axis=1)
        batch_input = batch_input[:, :, np.newaxis].copy()

        spatial_size = torch.LongTensor(batch_input.shape[:dimension])
        select_indices = tuple(locations.T)
        features = batch_input[select_indices]

        input_layer = scn.InputLayer(dimension, spatial_size, mode=3)
        # Facebook implementation applies only valid padding
        max_pool_layer = scn.MaxPooling(dimension, pool_size, pool_stride)

        output_layer = scn.SparseToDense(dimension=dimension, nPlanes=1)

        x = input_layer([torch.LongTensor(locations), torch.FloatTensor(features)])
        x = max_pool_layer(x)
        x = output_layer(x)

        correct_output = np.asarray([[1, 2],
                                     [0, 8]]).astype(float)
        self.assertListEqual(correct_output.tolist(), torch.squeeze(x).data.cpu().tolist())

    def test_sparse_maxpool(self):
        """
        Tests if the output of one layer asynchronous Max Pooling outputs expected values.
        """
        # Create Input
        dimension = 1
        pool_size = 3
        pool_stride = 2
        dense_input = np.array([0, -1, 0, 1, 2, 0, 1, 2, 3, 4, 0])
        locations = np.squeeze(dense_input).nonzero()[0]

        # Test with padding='valid'
        asyn_max_pool_layer_1 = ascn.asynMaxPool(dimension=dimension, filter_size=pool_size, filter_stride=pool_stride,
                                                 padding_mode='valid')
        asyn_output = asyn_max_pool_layer_1.forward(update_location=torch.tensor(locations)[:, None],
                                                    feature_map=torch.tensor(dense_input)[:, None])
        correct_output = [0, 2, 2, 3, 4]
        correct_indices = [2, 4, 4, 8, 9]

        self.assertListEqual(correct_output, torch.squeeze(asyn_output[1].float()).data.cpu().tolist())
        self.assertListEqual(correct_indices, torch.squeeze(asyn_output[2].float()).data.cpu().tolist())

    def test_fb_sparse_maxpool(self):
        """
        Tests if the output of one layer asynchronous Max Pooling outputs expected values.
        """
        for i_test in tqdm.tqdm(range(100)):
            # Create Input
            nIn = 4
            pool_size = 3
            pool_stride = 2
            spatial_dimensions = [5, 5]
            padding_mode = 'valid'
            dimension = len(spatial_dimensions)

            out = test_util.createInput(nIn=nIn, spatial_dimensions=spatial_dimensions, asynchronous_input=False,
                                        sequence_length=1, simplified=False)
            batch_input, batch_update_locations = out

            fb_max_layer, asyn_max_layer = test_util.createMaxLayers(dimension, facebook_layer=True,
                                                                     pool_size=pool_size, pool_stride=pool_stride,
                                                                     padding_mode=padding_mode)
            spatial_size = torch.LongTensor(batch_input.shape[:dimension])
            select_indices = tuple(batch_update_locations.T)
            features = batch_input[select_indices]
            input_layer = scn.InputLayer(dimension, spatial_size, mode=3)
            output_layer = scn.SparseToDense(dimension=dimension, nPlanes=nIn)

            # Facebook implementation
            fb_output = input_layer([torch.LongTensor(batch_update_locations), torch.FloatTensor(features)])
            fb_output = fb_max_layer(fb_output)
            fb_output = output_layer(fb_output)

            # Asynchronous sparse implementation
            asyn_output = asyn_max_layer.forward(update_location=torch.tensor(batch_update_locations),
                                                 feature_map=torch.tensor(batch_input))

            np_fb_output = np.squeeze(fb_output.cpu().numpy(), axis=0)
            fb_dim = np_fb_output.ndim
            np_fb_output = np_fb_output.transpose([i for i in range(1, fb_dim)] + [0])

            self.assertListEqual(np.squeeze(asyn_output[1].numpy()).tolist(),
                                 np.squeeze(np_fb_output).tolist())

    def test_asyn_sparse_maxpool(self):
        """
        Tests if the output of one layer asynchronous Max Pooling outputs expected values.
        """
        for i_test in tqdm.tqdm(range(100)):
            # Create Input
            nIn = 128
            pool_size = 3
            pool_stride = 2
            sequence_length = 20
            spatial_dimensions = [50, 50]
            padding_mode = 'valid'
            dimension = len(spatial_dimensions)

            out = test_util.createInput(nIn=nIn, spatial_dimensions=spatial_dimensions, asynchronous_input=True,
                                        sequence_length=sequence_length, simplified=False)
            batch_input, batch_update_locations, asyn_input, asyn_update_locations = out

            batch_max_layer, asyn_max_layer = test_util.createMaxLayers(dimension, facebook_layer=False,
                                                                        pool_size=pool_size, pool_stride=pool_stride,
                                                                        padding_mode=padding_mode)

            # Batch input
            batch_output = batch_max_layer.forward(update_location=torch.tensor(batch_update_locations),
                                                   feature_map=torch.tensor(batch_input))

            # Asynchronous sparse implementation
            for i_seq in range(sequence_length):
                asyn_output = asyn_max_layer.forward(update_location=torch.tensor(asyn_update_locations[i_seq]),
                                                     feature_map=torch.tensor(asyn_input[i_seq]))

            try:
                self.assertListEqual(np.squeeze(batch_output[1].numpy()).tolist(),
                                     np.squeeze(asyn_output[1].numpy()).tolist())
                self.assertListEqual(np.squeeze(batch_output[2].numpy()).tolist(),
                                     np.squeeze(asyn_output[2].numpy()).tolist())
            except AssertionError:
                print("Input")
                print(np.squeeze(batch_input))
                print("Batch Output")
                print(torch.squeeze(batch_output[1]))
                print("Batch Indices")
                print(torch.squeeze(batch_output[2]))
                print("-----")
                print("Asyn Output")
                print(torch.squeeze(asyn_output[1]))
                print("Asyn Indices")
                print(torch.squeeze(asyn_output[2]))

                raise AssertionError


if __name__ == '__main__':
    unittest.main()
