import os
import numpy as np
import torch
import torch.nn as nn
import logging

from data.image_prep import ImagePrep
from data.dataset import CustomDataset
from train import TrainModel
from test import TestModel
from ray import tune

if __name__ == '__main__':
    data_dir = os.path.abspath('images_structural')
    # ImagePrep(data_dir).save_images((64, 64))
    
    custom_dataset = CustomDataset(data_dir)
    custom_dataset.load_images()
    train_set, val_set, test_set = custom_dataset.train_val_test_split()
    
    config_train = {
        'lr': 1e-3,
        'batch_size': 6,
        'features': [64, 128, 256, 512],
        'epochs': 1,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    }
    config_optimize = {
        'lr': tune.choice([1e-5, 1e-3]),
        'batch_size': tune.choice([6, 12]),
        'features': tune.choice([[32, 64, 128, 256], [64, 128, 256, 512]]),
        'epochs': 1,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    }
    
    model = TrainModel(train_set, val_set, logging.DEBUG)
    # train_model = model.train_unet(config_train)
    best_model_config = model.optimize_unet(config_optimize, 1, 7, 1)
    
    test_model = TestModel(best_model_config, test_set, logging.DEBUG).test_best_model()
