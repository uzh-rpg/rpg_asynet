"""
Example command:
    python -m evaluation.sliding_window_flops --setting config/settings.yaml
    --save_dir "PATH_TO_DIR" --num_events 1 --num_samples 500
    --representation histogram  --use_multiprocessing
"""
import os
import argparse
import numpy as np
import torch
import tqdm
import random

from config.settings import Settings
from dataloader.dataset import getDataloader
from models.asyn_sparse_vgg import EvalAsynSparseVGGModel
from evaluation.reference_vgg_model import ReferenceVGGModel
from evaluation.compute_flops import computeFLOPS
from training.trainer import AbstractTrainer


np.random.seed(7)

class FlopEvaluation:
    def __init__(self, args, settings, save_dir='log/N_Caltech_FLOPs',):
        self.settings = settings
        self.save_dir = save_dir
        self.args = args
        self.multi_processing = args.use_multiprocessing
        self.compute_active_sites = args.compute_active_sites

        representation = args.representation or settings.event_representation
        if representation == 'histogram':
            self.nr_input_channels = 2
        elif representation == 'event_queue':
            self.nr_input_channels = 30
        else:
            raise NotImplementedError('Representation is not implemented')

        if settings.dataset_name == 'NCaltech101':
            self.layer_list = [['C', self.nr_input_channels, 16], ['BNRelu'], ['C', 16, 16], ['BNRelu'], ['MP'],
                               ['C', 16, 32], ['BNRelu'], ['C', 32, 32], ['BNRelu'], ['MP'],
                               ['C', 32, 64], ['BNRelu'], ['C', 64, 64], ['BNRelu'], ['MP'],
                               ['C', 64, 128], ['BNRelu'], ['C', 128, 128], ['BNRelu'], ['MP'],
                               ['C', 128, 256], ['BNRelu'], ['C', 256, 256], ['BNRelu'], ['MP'],
                               ['C', 256, 512], ['BNRelu'],
                               ['ClassicC', 512, 256, 3, 2], ['ClassicBNRelu'], ['ClassicFC', 256 * 3 * 2, 101]]
            self.nr_classes = 101
        elif settings.dataset_name == 'NCars':
            self.layer_list = [['C', self.nr_input_channels,    16], ['BNRelu'], ['C', 16, 16],   ['BNRelu'], ['MP'],
                               ['C', 16,   32], ['BNRelu'], ['C', 32, 32],   ['BNRelu'], ['MP'],
                               ['C', 32,   64], ['BNRelu'], ['C', 64, 64],   ['BNRelu'], ['MP'],
                               ['C', 64,  128], ['BNRelu'], ['C', 128, 128], ['BNRelu'], ['MP'],
                               ['C', 128, 256], ['BNRelu'],
                               ['ClassicC', 256, 128, 3, 2], ['ClassicBNRelu'], ['ClassicFC', 128*3*2, 101]]
            self.nr_classes = 2
        elif settings.dataset_name == 'NCaltech101_ObjectDetection' or settings.dataset_name == 'Prophesee':
            if self.settings.dataset_name == 'NCaltech101_ObjectDetection':
                out_put_map = 5 * 7
                self.nr_classes = 101
            else:
                out_put_map = 6 * 8
                self.nr_classes = 2
            self.layer_list = [['C', self.nr_input_channels, 16], ['BNRelu'], ['C', 16, 16],   ['BNRelu'], ['MP'],
                               ['C', 16,   32], ['BNRelu'], ['C', 32, 32],   ['BNRelu'], ['MP'],
                               ['C', 32,   64], ['BNRelu'], ['C', 64, 64],   ['BNRelu'], ['MP'],
                               ['C', 64,  128], ['BNRelu'], ['C', 128, 128], ['BNRelu'], ['MP'],
                               ['C', 128, 256], ['BNRelu'], ['C', 256, 256], ['BNRelu'],
                               ['ClassicC', 256, 256, 3, 2], ['ClassicBNRelu'],
                               ['ClassicFC', 256*out_put_map, 1024],  ['ClassicFC', 1024, self.nr_classes]]

        else:
            raise NotImplementedError('The specified dataset is not supported')

        self.train_loader = None
        self.nr_sampling_runs = None
        self.nr_events_timestep = None
        self.sliding_window_size = None
        self.dataloader = getDataloader(self.settings.dataset_name)

        self.device = torch.device("cpu")

    def evaluateFLOP(self):
        """Evaluate asynchronous spare VGG"""
        # nr_events_timestep = [i*1000 for i in range(1, 26)]
        self.nr_events_timestep = [25000, 25000+self.args.num_events]
        self.sliding_window_size = 25000
        self.nr_sampling_runs = self.args.num_samples

        nr_timesteps = len(self.nr_events_timestep)
        flops_overall = np.zeros([self.nr_sampling_runs, nr_timesteps, 5, len(self.layer_list)])
        flops_overall[:, :, 4, :] = np.array(self.nr_events_timestep)[np.newaxis, :, np.newaxis]

        if not self.multi_processing:
            for i_run in tqdm.tqdm(range(self.nr_sampling_runs)):
                out = self.evaluateSamplingRun(i_run)
                asyn_flops, ref_flops, sparse_flops, batch_flops = out
                flops_overall[i_run, :, 0, :] = asyn_flops
                flops_overall[i_run, :, 1, :] = ref_flops
                flops_overall[i_run, :, 2, :] = sparse_flops
                flops_overall[i_run, :, 3, :] = batch_flops

                string = "FLOPS: \n\tasyn sparse: %s  \n\tasyn dense: %s    \n\tbatch sparse: %s    \n\tdense batch: %s"
                print(string % (
                    asyn_flops.sum(-1)[-1],
                    ref_flops.sum(-1)[-1],
                    sparse_flops.sum(-1)[-1],
                    batch_flops.sum(-1)[-1],
                ))
        else:
            import multiprocessing
            pool = multiprocessing.Pool(processes=(os.cpu_count() - 2))
            # pool = multiprocessing.Pool(processes=2)
            result = pool.map(self.evaluateSamplingRun, range(self.nr_sampling_runs))
            result = np.asarray(result).transpose([0, 2, 1, 3])
            flops_overall[:, :, :4, :] = result

        flops_overall_std = flops_overall.sum(-1).std(0)
        flops_overall = flops_overall.mean(0)

        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        if self.compute_active_sites:
            np.save(os.path.join(self.save_dir, 'ActiveSites'), flops_overall)
            np.save(os.path.join(self.save_dir, 'ActiveSites_std'), flops_overall_std)
        else:

            np.save(os.path.join(self.save_dir, 'NCaltech_FLOPs'), flops_overall)
            np.save(os.path.join(self.save_dir, 'NCaltech_FLOPs_std'), flops_overall_std)

    def evaluateSamplingRun(self, i_run):
        if self.settings.dataset_name == 'NCaltech101':
            spatial_dimensions = [191, 255]
        elif self.settings.dataset_name == 'NCars':
            spatial_dimensions = [95, 127]
        elif self.settings.dataset_name == 'NCaltech101_ObjectDetection':
            spatial_dimensions = [191, 255]
        elif self.settings.dataset_name == 'Prophesee':
            spatial_dimensions = [223, 287]

        # ---- Create Models ----
        asyn_model = EvalAsynSparseVGGModel(nr_classes=self.nr_classes, layer_list=self.layer_list, device=self.device,
                                            input_channels=self.nr_input_channels)
        asyn_model.convertWeightsDouble()
        ref_model = ReferenceVGGModel(self.layer_list, device=self.device)

        # ---- Create Input -----
        nr_timestep = len(self.nr_events_timestep)
        train_dataset = self.dataloader(self.settings.dataset_path, 'all', self.settings.height,
                                        self.settings.width, augmentation=False, mode='validation',
                                        nr_events_window=-1, shuffle=True)
        data_idx = int(i_run % train_dataset.__len__())
        print("using data index: %s" % data_idx)
        events, labels, histogram = train_dataset.__getitem__(data_idx)
        nr_events = events.shape[0]
        random.seed(7)
        start_point = random.randrange(0, max(1, nr_events - self.nr_events_timestep[-1]))
        events = events[start_point:, :]
        input_timesteps = self.createInputs(events, self.nr_events_timestep, spatial_dimensions,
                                            self.sliding_window_size, asyn_model.generateAsynInput)

        asyn_flops = np.zeros([nr_timestep, len(self.layer_list)])
        ref_flops = np.zeros([nr_timestep, len(self.layer_list)])
        sparse_flops = np.zeros([nr_timestep, len(self.layer_list)])
        batch_flops = np.zeros([nr_timestep, len(self.layer_list)])

        if self.multi_processing and i_run % (os.cpu_count() + 2) == 0:
            pbar = tqdm.tqdm(total=nr_timestep, unit='Batch Event', unit_scale=True)

        with torch.no_grad():
            for i_sequence, nr_events in enumerate(self.nr_events_timestep):
                # Batch sparse model needs to be reinitialised
                sparse_model = EvalAsynSparseVGGModel(nr_classes=self.nr_classes, layer_list=self.layer_list,
                                                      device=self.device, input_channels=self.nr_input_channels)
                sparse_model.convertWeightsDouble()

                x_asyn = [None] * 5
                x_asyn[0] = input_timesteps[1][i_sequence][:, :2].to(self.device)
                x_asyn[1] = input_timesteps[0][i_sequence, :, :, :].to(self.device)

                nr_layers = len(self.layer_list)
                asyn_active_sites = np.zeros([nr_layers])
                asyn_nr_sites = np.zeros([nr_layers])

                sparse_active_sites = np.zeros([nr_layers])
                sparse_nr_sites = np.zeros([nr_layers])
                ref_active_sites = np.zeros([nr_layers])
                x_sparse = self.createSparseInput(input_timesteps[0][i_sequence, :, :, :])
                ref_model.forward((input_timesteps[2][i_sequence, :, :, :]**2).to(self.device), ref_active_sites)

                if self.compute_active_sites:
                    _ = asyn_model.forward(x_asyn, asyn_active_sites, asyn_nr_sites, flop_calculation=False)
                    asyn_flops[i_sequence] = asyn_active_sites
                    batch_flops[i_sequence] = asyn_nr_sites
                    _ = sparse_model.forward(x_sparse, sparse_active_sites, sparse_nr_sites, flop_calculation=False)
                    ref_flops[i_sequence] = ref_active_sites
                    sparse_flops[i_sequence] = sparse_active_sites
                else:
                    _ = asyn_model.forward(x_asyn, asyn_active_sites, asyn_nr_sites, flop_calculation=True)

                    asyn_flops[i_sequence] = computeFLOPS(asyn_active_sites, self.layer_list, sparse_model=True,
                                                          layerwise=True)
                    batch_flops[i_sequence] = computeFLOPS(asyn_nr_sites, self.layer_list, sparse_model=False,
                                                           layerwise=True)

                    _ = sparse_model.forward(x_sparse, sparse_active_sites, sparse_nr_sites, flop_calculation=True)
                    ref_flops[i_sequence] = computeFLOPS(ref_active_sites, self.layer_list, sparse_model=False,
                                                         layerwise=True)
                    sparse_flops[i_sequence] = computeFLOPS(sparse_active_sites, self.layer_list, sparse_model=True,
                                                            layerwise=True)

                if self.multi_processing and i_run % (os.cpu_count() + 2) == 0:
                    pbar.update(1)

                # print('--------Sequence %s----------' % i_sequence)
        if self.multi_processing and i_run % (os.cpu_count() + 2) == 0:
            pbar.close()

        if self.multi_processing:
            if not os.path.isdir(self.save_dir):
                os.makedirs(self.save_dir)
            timestep_array = np.tile(np.array(self.nr_events_timestep)[:, np.newaxis, np.newaxis],
                                     [1, 1, len(self.layer_list)])
            container = np.concatenate([asyn_flops[:, np.newaxis, :],
                                        ref_flops[:, np.newaxis, :],
                                        sparse_flops[:, np.newaxis, :],
                                        batch_flops[:, np.newaxis, :],
                                        timestep_array], axis=1)
            np.save(os.path.join(self.save_dir, 'NCaltech_FLOPs_run_' + str(i_run)), container)

        return asyn_flops, ref_flops, sparse_flops, batch_flops

    def createInputs(self, events, nr_events_timestep, spatial_dimensions, sliding_window_size, fn_generateAsynInput):
        """Creates for each timestep the input according to sliding window histogram"""
        start_windows = [nr_events - sliding_window_size for nr_events in nr_events_timestep]
        changing_timesteps = start_windows + nr_events_timestep
        changing_timesteps.sort()

        tensor_spatial_dimensions = torch.tensor(spatial_dimensions)
        change_histogram = torch.zeros([len(changing_timesteps)-1] + spatial_dimensions + [self.nr_input_channels])

        for i_change in range(len(changing_timesteps) - 1):
            nr_changing_events = changing_timesteps[i_change+1] - changing_timesteps[i_change]
            batch_events = events[changing_timesteps[i_change]:changing_timesteps[i_change+1]]
            update_locations, new_histogram = fn_generateAsynInput(batch_events, tensor_spatial_dimensions,
                                                                   original_shape=[self.settings.height,
                                                                                   self.settings.width])
            # As input image dimension is upsampled, the number of new events can be increased as well.
            np.random.seed(7)
            random_permutation = np.random.permutation(update_locations.shape[0])
            idx_discard = random_permutation[:-nr_changing_events]
            new_histogram[update_locations[idx_discard, 0],
                          update_locations[idx_discard, 1], :] = torch.tensor([0, 0]).float()
            # update_locations = update_locations[random_permutation[-nr_changing_events:], :]

            change_histogram[i_change, :, :, :2] = new_histogram

        new_histogram = torch.zeros([len(nr_events_timestep)] + spatial_dimensions + [self.nr_input_channels])
        input_histogram = torch.zeros([len(nr_events_timestep)] + spatial_dimensions + [self.nr_input_channels])
        input_update_locations = []

        for i_timestep in range(len(nr_events_timestep)):
            if nr_events_timestep[i_timestep] - start_windows[i_timestep] < 0:
                raise ValueError('Sliding window is not full. Change nr_events_timestep')
            start_idx_changing = changing_timesteps.index(start_windows[i_timestep])
            end_idx_changing = changing_timesteps.index(nr_events_timestep[i_timestep])

            input_histogram[i_timestep] = change_histogram[start_idx_changing:end_idx_changing, :, :, :].sum(0)
            # if i_timestep=0, the input_histogram[i_timestep - 1] = input_histogram[-1], which is zero
            new_histogram[i_timestep] = input_histogram[i_timestep] - input_histogram[i_timestep-1]

            update_locations, _ = AbstractTrainer.denseToSparse(torch.tensor(new_histogram[i_timestep].unsqueeze(0)**2,
                                                                             requires_grad=False))

            # Catch cases, where the input is downsampled and no update locations are found
            if update_locations.shape[0] <= 1 and self.settings.dataset_name == 'Prophesee':
                # Use final event as location
                nr_events_insert = 2 - update_locations.shape[0]
                end_event_idx = nr_events_timestep[i_timestep] + 2
                add_events = events[end_event_idx:(end_event_idx + nr_events_insert)].astype(np.float)
                add_events[:, 0] *= spatial_dimensions[1] / self.settings.width
                add_events[:, 1] *= spatial_dimensions[0] / self.settings.height
                add_events = np.floor(add_events).astype(np.int)
                update_locations = torch.cat([update_locations, torch.zeros([nr_events_insert, 3], dtype=torch.long)],
                                             dim=0)
                update_locations[-nr_events_insert:, 0] = torch.from_numpy(add_events[:, 1])
                update_locations[-nr_events_insert:, 1] = torch.from_numpy(add_events[:, 0])

            input_update_locations.append(update_locations)

        return input_histogram, input_update_locations, new_histogram

    def createSparseInput(self, input_histogram):
        update_locations, features = AbstractTrainer.denseToSparse(input_histogram.unsqueeze(0))
        x_sparse = [None] * 5
        x_sparse[0] = update_locations[:, :2].to(self.device)
        x_sparse[1] = input_histogram.to(self.device)

        return x_sparse

    def printLastStepFLOPs(self):
        if self.compute_active_sites:
            flops_overall = np.load(os.path.join(self.save_dir, 'ActiveSites.npy'))
            flops_overall_std = np.load(os.path.join(self.save_dir, 'ActiveSites_std.npy'))
        else:
            flops_overall = np.load(os.path.join(self.save_dir, 'NCaltech_FLOPs.npy'))
            flops_overall_std = np.load(os.path.join(self.save_dir, 'NCaltech_FLOPs_std.npy'))

        nr_layers = flops_overall.shape[-1]
        if flops_overall.ndim == 3:
            flops_overall = flops_overall.sum(-1)
        flops_overall[:, -1] /= nr_layers
        modes = ['Asyn Sparse', 'Asyn Conventional', 'Batch Sparse', 'Batch Conventional']

        string = '#Total Length Window          :   {:.0f}\n' \
                 '#Events for Update            :   {:.0f}\n' \
                 '===================================='.format(flops_overall[0, -1],
                                                               flops_overall[1, -1] - flops_overall[0, -1])
        print(string)

        if self.compute_active_sites:
            string = 'Mode:  {}                               \n' \
                     '#FLOPs for Update :   {:.0f} +- {:.0f}  \n' \
                     '====================================\n'
        else:
            string = 'Mode:  {}                               \n' \
                     '#Active Sites      :   {:.0f} +- {:.0f}  \n' \
                     '====================================\n'
        for i, mode in enumerate(modes):
            print(string.format(mode, flops_overall[1, i], flops_overall_std[1, i]))


def main():
    parser = argparse.ArgumentParser(description='Evaluate network.')
    parser.add_argument('--settings_file', help='Path to settings yaml', required=True)
    parser.add_argument('--save_dir', help='Path to save location', required=True)
    parser.add_argument('--num_events', default=1, type=int)
    parser.add_argument('--num_samples', default=500, type=int)
    parser.add_argument('--representation', default="")
    parser.add_argument('--use_multiprocessing', help='If multiprocessing should be used', action='store_true')
    parser.add_argument('--compute_active_sites', help='If active sites should be calculated', action='store_true')

    args = parser.parse_args()
    settings_filepath = args.settings_file
    save_dir = args.save_dir

    settings = Settings(settings_filepath, generate_log=False)

    evaluator = FlopEvaluation(args, settings, save_dir=save_dir)
    evaluator.evaluateFLOP()
    evaluator.printLastStepFLOPs()


if __name__ == "__main__":
    main()
