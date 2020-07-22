import os
import time
import yaml
import torch
import shutil


class Settings:
    def __init__(self, settings_yaml, generate_log=True):
        assert os.path.isfile(settings_yaml), settings_yaml

        with open(settings_yaml, 'r') as stream:
            settings = yaml.load(stream, yaml.Loader)

            # --- hardware ---
            hardware = settings['hardware']
            gpu_device = hardware['gpu_device']

            self.gpu_device = torch.device("cpu") if gpu_device == "cpu" else torch.device("cuda:" + str(gpu_device))

            self.num_cpu_workers = hardware['num_cpu_workers']
            if self.num_cpu_workers < 0:
                self.num_cpu_workers = os.cpu_count()

            # --- Model ---
            model = settings['model']
            self.model_name = model['model_name']

            # --- dataset ---
            dataset = settings['dataset']
            self.dataset_name = dataset['name']
            self.event_representation = dataset['event_representation']
            if self.dataset_name == 'NCaltech101':
                dataset_specs = dataset['ncaltech101']
            elif self.dataset_name == 'NCaltech101_ObjectDetection':
                dataset_specs = dataset['ncaltech101_objectdetection']
            elif self.dataset_name == 'Prophesee':
                dataset_specs = dataset['prophesee']
            elif self.dataset_name == 'NCars':
                dataset_specs = dataset['ncars']

            self.dataset_path = dataset_specs['dataset_path']
            assert os.path.isdir(self.dataset_path)
            self.object_classes = dataset_specs['object_classes']
            self.height = dataset_specs['height']
            self.width = dataset_specs['width']
            self.nr_events_window = dataset_specs['nr_events_window']

            # --- checkpoint ---
            checkpoint = settings['checkpoint']
            self.resume_training = checkpoint['resume_training']
            assert isinstance(self.resume_training, bool)
            self.resume_ckpt_file = checkpoint['resume_file']
            self.use_pretrained = checkpoint['use_pretrained']
            self.pretrained_dense_vgg = checkpoint['pretrained_dense_vgg']
            self.pretrained_sparse_vgg = checkpoint['pretrained_sparse_vgg']

            # --- directories ---
            directories = settings['dir']
            log_dir = directories['log']

            # --- logs ---
            if generate_log:
                timestr = time.strftime("%Y%m%d-%H%M%S")
                log_dir = os.path.join(log_dir, timestr)
                os.makedirs(log_dir)
                settings_copy_filepath = os.path.join(log_dir, 'settings.yaml')
                shutil.copyfile(settings_yaml, settings_copy_filepath)
                self.ckpt_dir = os.path.join(log_dir, 'checkpoints')
                os.mkdir(self.ckpt_dir)
                self.vis_dir = os.path.join(log_dir, 'visualization')
                os.mkdir(self.vis_dir)
            else:
                self.ckpt_dir = os.path.join(log_dir, 'checkpoints')
                self.vis_dir = os.path.join(log_dir, 'visualization')

            # --- optimization ---
            optimization = settings['optim']
            self.batch_size = optimization['batch_size']
            self.init_lr = float(optimization['init_lr'])
            self.steps_lr = optimization['steps_lr']
            self.factor_lr = float(optimization['factor_lr'])
