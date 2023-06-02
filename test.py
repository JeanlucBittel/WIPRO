import numpy as np
import torch.nn as nn
import logging
import torch

from util.logger import CustomLogger
from util.plotter import Plotter
from models.unet import UNET
from monai.transforms import Compose
from data.data_loader import NIfTIDataset
from torchmetrics import StructuralSimilarityIndexMeasure, MeanAbsolutePercentageError, MeanSquaredError
from torch.utils.data import DataLoader

class TestModel():
    
    def __init__(self, model_config, test_set, log_level=logging.WARNING):
        self.model_config = model_config
        self.test_set = test_set
        self.log_level = log_level
        self.log = CustomLogger('U-Net Logger').setLogLevel(self.log_level)
        self.test_transforms = Compose([])
        self.test_transforms_flair = Compose([])
        self.test_transforms_mask = Compose([])
        self.test_ds = NIfTIDataset(self.test_set, [self.test_transforms, self.test_transforms_flair, self.test_transforms_mask], (64, 64))

    def test_best_model(self):
        device = self.model_config.config['device']
        batch_size = self.model_config.config['batch_size']

        best_trained_model = UNET(in_channels=3, out_channels=1, features=self.model_config.config['features']).to(device)

        test_loader = DataLoader(self.test_ds, batch_size=batch_size, num_workers=1)

        best_checkpoint = self.model_config.checkpoint.to_air_checkpoint()
        best_checkpoint_data = best_checkpoint.to_dict()
        best_trained_model.load_state_dict(best_checkpoint_data["model_state_dict"])

        ssim_loss = StructuralSimilarityIndexMeasure().to(device)
        mse_loss = nn.MSELoss().to(device)
        mae_loss = nn.L1Loss().to(device)
        mape_loss = MeanAbsolutePercentageError().to(device)
        rmse_loss = MeanSquaredError(squared=False).to(device)

        test_loss_mse = np.array([])
        test_loss_mae = np.array([])

        # Testing
        batches_trained = 0
        running_loss = 0.0
        baseline_running_loss = 0.0
        ssim_sum = 0
        ssim_base_sum = 0
        mse_sum = 0
        mse_base_sum = 0
        mae_sum = 0
        mae_base_sum = 0
        mape_sum = 0
        mape_base_sum = 0
        rmse_sum = 0
        rmse_base_sum = 0

        with torch.no_grad():
            for times, (inputs, labels, masks) in enumerate(test_loader, 1):
                (inputs, labels, masks) = (inputs.to(device), labels.to(device), masks.to(device))
                test_len = len(test_loader.dataset) * inputs.shape[-1]

                for i in range(inputs.shape[-1]):
                    batches_trained += len(inputs)

                    inputs_s = inputs[:,:,:,:,i].reshape(len(inputs), 3, 64, 64)
                    masks_s = masks[:,:,:,:,i].reshape(len(inputs), 1, 64, 64)
                    labels_s = labels[:,:,:,:,i].reshape(len(inputs), 1, 64, 64)

                    output = best_trained_model(inputs_s)
                    output *= masks_s

                    baseline = masks_s * 0.5

                    ssim = ssim_loss(output, labels_s)
                    ssim_sum += ssim
                    ssim_base = ssim_loss(baseline, labels_s)
                    ssim_base_sum += ssim_base
                    mse = mse_loss(output, labels_s)
                    mse_sum += mse
                    mse_base = mse_loss(baseline, labels_s)
                    mse_base_sum += mse_base
                    mae = mae_loss(output, labels_s)
                    mae_sum += mae
                    mae_base = mae_loss(baseline, labels_s)
                    mae_base_sum += mae_base
                    mape = mape_loss(output, labels_s)
                    mape_sum += mape
                    mape_base = mape_loss(baseline, labels_s)
                    mape_base_sum += mape_base
                    rmse = rmse_loss(output, labels_s)
                    rmse_sum += rmse
                    rmse_base = rmse_loss(baseline, labels_s)
                    rmse_base_sum += rmse_base

                    # Show progress
                    self.log.info('Testing: %d/%d (%.0f%%) Image: %d\nMSE: %.3f/%.3f\tMAE: %.3f/%.3f\tMAPE: %.3f/%.3f\tRMSE: %.3f/%.3f\tSSIM: %.3f/%.3f\n',
                              batches_trained, test_len, 100. * batches_trained / test_len, i, mse, mse_base, mae, mae_base, mape, mape_base, rmse, rmse_base, ssim, ssim_base)

            times *= inputs.shape[-1]
            test_loss_mse = np.append(test_loss_mse, mse_sum.cpu() / times)
            test_loss_mae = np.append(test_loss_mae, mae_sum.cpu() / times)
            self.log.info('Average MSE: %.3f/%.3f\tAverage MAE: %.3f/%.3f\tAverage MAPE: %.3f/%.3f\tAverage RMSE: %.3f/%.3f\tAverage SSIM: %.3f/%.3f', 
                      mse_sum / times, mse_base_sum / times, mae_sum / times, mae_base_sum / times, mape_sum / times, mape_base_sum / times, rmse_sum / times, rmse_base_sum / times, ssim_sum / times, ssim_base_sum / times)

            img_label = labels[-1].cpu().detach() 
            img_input = inputs[-1].cpu().detach() 
            img_mask = masks[-1].cpu().detach() 
            img_output = output[-1].cpu().detach().numpy()

            self.log.info('Testing Image')
            Plotter.plot_images(img_input, img_label, img_mask, img_output)        