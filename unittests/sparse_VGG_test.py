"""
Example command: python -m unittests.sparse_VGG_test
"""
import numpy as np
import torch
import tqdm
import unittest

from dataloader.dataset import NCaltech101
from models.asyn_sparse_vgg import asynSparseVGG
from models.facebook_sparse_vgg import FBSparseVGG
from training.trainer import AbstractTrainer
import utils.test_util as test_util

# DEVICE = torch.device("cuda:0")
DEVICE = torch.device("cpu")


class TestSparseVGG(unittest.TestCase):
    def test_sparse_VGG_artificial_input(self):
        """Tests if output of sparse VGG is equivalent to the facebook implementation"""
        print('Asynchronous N-Layer Test')
        for i_test in tqdm.tqdm(range(1)):
            # print('Test: %s' % i_test)
            # print('#######################')
            # print('#       New Test      #')
            # print('#######################')
            nr_classes = 101
            sequence_length = 10

            # ---- Facebook VGG ----
            fb_model = FBSparseVGG(nr_classes).eval()

            pth = 'PATH_TO_MODEL'
            fb_model.load_state_dict(torch.load(pth, map_location={'cuda:0': 'cpu'})['model'])

            # ---- Create Input ----
            out = test_util.createInput(nIn=2, spatial_dimensions=[191, 255],
                                        asynchronous_input=True, sequence_length=sequence_length, simplified=False)
            batch_input, batch_update_locations, asyn_input, asyn_update_locations = out

            # ---- Asynchronous VGG ----
            layer_list = [['C', 2,    16], ['BNRelu'], ['C', 16, 16],   ['BNRelu'], ['MP'],
                          ['C', 16,   32], ['BNRelu'], ['C', 32, 32],   ['BNRelu'], ['MP'],
                          ['C', 32,   64], ['BNRelu'], ['C', 64, 64],   ['BNRelu'], ['MP'],
                          ['C', 64,  128], ['BNRelu'], ['C', 128, 128], ['BNRelu'], ['MP'],
                          ['C', 128, 256], ['BNRelu'], ['C', 256, 256], ['BNRelu'], ['MP'],
                          ['C', 256, 512], ['BNRelu'],
                          ['ClassicC', 512, 256, 3, 2], ['ClassicBNRelu'], ['ClassicFC', 256*3*2, 2]]

            asyn_model = asynSparseVGG(nr_classes=101, layer_list=layer_list, device=DEVICE)
            asyn_model.setWeightsEqual(fb_model)

            # ---- Facebook VGG ----
            select_indices = tuple(batch_update_locations.T)
            features = batch_input[select_indices]
            batch_locations = torch.tensor(batch_update_locations, dtype=torch.long)
            features = torch.tensor(features, dtype=torch.float)

            fb_output = fb_model([batch_locations, features])

            # ---- Asynchronous VGG ----
            with torch.no_grad():
                for i_sequence in range(sequence_length):
                    x_asyn = [None] * 5
                    x_asyn[0] = torch.tensor(asyn_update_locations[i_sequence], dtype=torch.long).to(DEVICE)
                    x_asyn[1] = torch.tensor(asyn_input[i_sequence], dtype=torch.float).to(DEVICE)

                    asyn_output = asyn_model.forward(x_asyn)
                    print('--------Sequence %s----------' % i_sequence)

            if fb_output.ndim == 4:
                np.testing.assert_almost_equal(asyn_output[1].float().data.cpu().numpy(),
                                               fb_output.squeeze(0).detach().numpy().transpose(1, 2, 0), decimal=5)
            else:
                np.testing.assert_almost_equal(asyn_output[1].float().data.cpu().numpy(),
                                               fb_output.squeeze(0).detach().numpy(), decimal=5)

    def test_sparse_VGG_event_input(self):
        """Tests if output of sparse VGG is equivalent to the facebook implementation"""
        print('Asynchronous N-Layer Test')
        nr_classes = 101
        nr_last_events = 1000
        sequence_length = 10

        # ---- Facebook VGG ----
        fb_model = FBSparseVGG(nr_classes).eval()
        spatial_dimensions = fb_model.spatial_size

        # ---- Create Input ----
        dataset_path = 'PATH_TO_DATA/N-Caltech101'
        height = 180
        width = 240
        train_dataset = NCaltech101(dataset_path, ['Motorbikes'], height, width, augmentation=False,
                                    mode='validation', nr_events_window=nr_last_events)

        events, labels, histogram = train_dataset.__getitem__(0)

        histogram = torch.from_numpy(histogram[np.newaxis, :, :])
        histogram = torch.nn.functional.interpolate(histogram.permute(0, 3, 1, 2), torch.Size(spatial_dimensions))
        histogram = histogram.permute(0, 2, 3, 1)
        locations, features = AbstractTrainer.denseToSparse(histogram)

        # ---- Facebook VGG ----
        pth = 'PATH_TO_MODEL'
        fb_model.load_state_dict(torch.load(pth, map_location={'cuda:0': 'cpu'})['model'])

        fb_output = fb_model([locations, features])

        # ---- Asynchronous VGG ----
        layer_list = [['C', 2,    16], ['BNRelu'], ['C', 16, 16],   ['BNRelu'], ['MP'],
                      ['C', 16,   32], ['BNRelu'], ['C', 32, 32],   ['BNRelu'], ['MP'],
                      ['C', 32,   64], ['BNRelu'], ['C', 64, 64],   ['BNRelu'], ['MP'],
                      ['C', 64,  128], ['BNRelu'], ['C', 128, 128], ['BNRelu'], ['MP'],
                      ['C', 128, 256], ['BNRelu'], ['C', 256, 256], ['BNRelu'], ['MP'],
                      ['C', 256, 512], ['BNRelu'],
                      ['ClassicC', 512, 256, 3, 2], ['ClassicBNRelu'], ['ClassicFC', 256*3*2, 2]]
        asyn_model = asynSparseVGG(nr_classes=101, layer_list=layer_list, device=DEVICE)
        asyn_model.setWeightsEqual(fb_model)

        list_spatial_dimensions = [spatial_dimensions.cpu().numpy()[0], spatial_dimensions.cpu().numpy()[1]]
        input_histogram = torch.zeros(list_spatial_dimensions + [2])
        step_size = nr_last_events // sequence_length

        with torch.no_grad():
            for i_sequence in range(sequence_length):
                new_batch_events = events[(step_size*i_sequence):(step_size*(i_sequence + 1)), :]
                update_locations, new_histogram = asyn_model.generateAsynInput(new_batch_events, spatial_dimensions,
                                                                               original_shape=[height, width])
                input_histogram = input_histogram + new_histogram
                x_asyn = [None] * 5
                x_asyn[0] = update_locations[:, :2].to(DEVICE)
                x_asyn[1] = input_histogram.to(DEVICE)

                asyn_output = asyn_model.forward(x_asyn)
                print('--------Sequence %s----------' % i_sequence)

        if fb_output.ndim == 4:
            np.testing.assert_almost_equal(asyn_output[1].float().data.cpu().numpy(),
                                           fb_output.squeeze(0).detach().numpy().transpose(1, 2, 0), decimal=5)
        else:
            np.testing.assert_almost_equal(asyn_output[1].float().data.cpu().numpy(),
                                           fb_output.squeeze(0).detach().numpy(), decimal=5)


if __name__ == '__main__':
    unittest.main()
