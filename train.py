"""
Example usage: CUDA_VISIBLE_DEVICES=1, python train.py --settings_file "config/settings.yaml"
"""
import argparse

from config.settings import Settings
from training.trainer import FBSparseVGGModel
from training.trainer import DenseVGGModel
from training.trainer import SparseObjectDetModel
from training.trainer import DenseObjectDetModel


def main():
    parser = argparse.ArgumentParser(description='Train network.')
    parser.add_argument('--settings_file', help='Path to settings yaml', required=True)

    args = parser.parse_args()
    settings_filepath = args.settings_file

    settings = Settings(settings_filepath, generate_log=True)

    if settings.model_name == 'fb_sparse_vgg':
        trainer = FBSparseVGGModel(settings)
    elif settings.model_name == 'dense_vgg':
        trainer = DenseVGGModel(settings)
    elif settings.model_name == 'fb_sparse_object_det':
        trainer = SparseObjectDetModel(settings)
    elif settings.model_name == 'dense_object_det':
        trainer = DenseObjectDetModel(settings)
    else:
        raise ValueError('Model name %s specified in the settings file is not implemented' % settings.model_name)

    trainer.train()


if __name__ == "__main__":
    main()
