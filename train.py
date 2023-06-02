import logging
import numpy as np
import torch
import torch.nn as nn

from util.logger import CustomLogger
from util.plotter import Plotter
from monai.transforms import Compose, RandAffine
from data.data_loader import NIfTIDataset
from models.unet import UNET
from torch.utils.data import DataLoader
from torchmetrics import StructuralSimilarityIndexMeasure, MeanAbsolutePercentageError, MeanSquaredError
from ray import tune
from ray.air import session, Checkpoint
from ray.tune.schedulers import ASHAScheduler
from functools import partial

class TrainModel():
    
    def __init__(self, train_set, val_set, log_level=logging.WARNING):
        self.train_set = train_set
        self.val_set = val_set
        self.log_level = log_level
        self.log = CustomLogger('U-Net Logger').setLogLevel(self.log_level)
        
        degrees = 10*np.pi/180
        shear_range = (0.15, 0.15, 0)
        scale_range = (0.15, 0.15, 0.15)

        self.train_transforms = Compose([
            RandAffine(prob=0.3, padding_mode='zeros', rotate_range=(0, 0, degrees), mode='bilinear'),
            RandAffine(prob=0.3, padding_mode='zeros', shear_range=shear_range, mode='bilinear'),
            RandAffine(prob=0.3, padding_mode='zeros', scale_range=scale_range, mode='bilinear')
        ])
        self.train_transforms_flair = Compose([
            RandAffine(prob=0.3, padding_mode='zeros', rotate_range=(0, 0, degrees), mode='bilinear'),
            RandAffine(prob=0.3, padding_mode='zeros', shear_range=shear_range, mode='bilinear'),
            RandAffine(prob=0.3, padding_mode='zeros', scale_range=scale_range, mode='bilinear')
        ])
        self.train_transforms_mask = Compose([
            RandAffine(prob=0.3, padding_mode='zeros', rotate_range=(0, 0, degrees), mode='nearest'),
            RandAffine(prob=0.3, padding_mode='zeros', shear_range=shear_range, mode='nearest'),
            RandAffine(prob=0.3, padding_mode='zeros', scale_range=scale_range, mode='nearest')
        ])

        self.val_transforms = Compose([])
        self.val_transforms_flair = Compose([])
        self.val_transforms_mask = Compose([])

        self.train_ds = NIfTIDataset(self.train_set, [self.train_transforms, self.train_transforms_flair, self.train_transforms_mask], (64, 64))
        self.val_ds = NIfTIDataset(self.val_set, [self.val_transforms, self.val_transforms_flair, self.val_transforms_mask], (64, 64))
        
    def train_unet(self, config, optimize=False):
        epochs = config['epochs']
        device = config['device']
        batch_size = config['batch_size']
        
        model = UNET(in_channels=3, out_channels=1, features=config['features']).to(device)
        
        last_loss = 100
        patience = 5
        trigger_times = 0
        start_epoch = 1

        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        mae_loss = nn.L1Loss().to(device)
        mse_loss = nn.MSELoss().to(device)
        mape_loss = MeanAbsolutePercentageError().to(device)
        rmse_loss = MeanSquaredError(squared=False).to(device)
        ssim_loss = StructuralSimilarityIndexMeasure().to(device)

        train_loss_mse = np.array([])
        train_loss_mae = np.array([])
        val_loss_mse = np.array([])
        val_loss_mae = np.array([])

        if optimize:
            checkpoint = session.get_checkpoint()

            if checkpoint:
                checkpoint_state = checkpoint.to_dict()
                start_epoch = checkpoint_state["epoch"]
                model.load_state_dict(checkpoint_state["model_state_dict"])
                optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])

        train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, num_workers=1)
        val_loader = DataLoader(self.val_ds, batch_size=batch_size, num_workers=1)

        for epoch in range(start_epoch, epochs + 1):
            # Training
            if self.log_level <= logging.WARNING:
                self.log.info('-'*50)
                self.log.info('Training')
                self.log.info('-'*50)
                self.log.info('Train Epoch: %d/%d', epoch, epochs)

            model.train()
            batches_trained = 0
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

            for times, (inputs, labels, masks) in enumerate(train_loader, 1):
                (inputs, labels, masks) = (inputs.to(device), labels.to(device), masks.to(device))
                train_len = len(train_loader.dataset) * inputs.shape[-1]

                for i in range(inputs.shape[-1]):
                    batches_trained += len(inputs)

                    inputs_s = inputs[:,:,:,:,i].reshape(len(inputs), 3, 64, 64)
                    masks_s = masks[:,:,:,:,i].reshape(len(inputs), 1, 64, 64)
                    labels_s = labels[:,:,:,:,i].reshape(len(inputs), 1, 64, 64)

                    # Zero the gradients
                    optimizer.zero_grad()

                    # Forward and backward propagation            
                    output = model(inputs_s)
                    output *= masks_s

                    mae = mae_loss(output, labels_s)
                    mae_sum += mae.item()

                    mae.backward()
                    optimizer.step()

                    baseline = masks_s * 0.5
                    mae_base = mae_loss(baseline, labels_s)
                    mae_base_sum += mae_base.item()

                    ssim = ssim_loss(output, labels_s)
                    ssim_sum += ssim
                    ssim_base = ssim_loss(baseline, labels_s)
                    ssim_base_sum += ssim_base
                    mse = mse_loss(output, labels_s)
                    mse_sum += mse
                    mse_base = mse_loss(baseline, labels_s)
                    mse_base_sum += mse_base
                    mape = mape_loss(output, labels_s)
                    mape_sum += mape
                    mape_base = mape_loss(baseline, labels_s)
                    mape_base_sum += mape_base
                    rmse = rmse_loss(output, labels_s)
                    rmse_sum += rmse
                    rmse_base = rmse_loss(baseline, labels_s)
                    rmse_base_sum += rmse_base

                    # Show progress
                    self.log.debug('Train Epoch: %d/%d [%d/%d (%.0f%%)] Image: %d\nSSIM: %.3f/%.3f\tMSE: %.3f/%.3f\tMAE: %.3f/%.3f\tMAPE: %.3f/%.3f\tRMSE: %.3f/%.3f\n',
                              epoch, epochs, batches_trained, train_len, 100. * batches_trained / train_len, i, ssim, ssim_base, mse, mse_base, mae, mae_base, mape, mape_base, rmse, rmse_base)

            times *= inputs.shape[-1]
            train_loss_mse = np.append(train_loss_mse, mse_sum / times)
            train_loss_mae = np.append(train_loss_mae, mae_sum / times)
            self.log.debug('Average SSIM: %.3f/%.3f\tAverage MSE: %.3f/%.3f\tAverage MAE: %.3f/%.3f\tAverage MAPE: %.3f/%.3f\tAverage RMSE: %.3f/%.3f', 
                      ssim_sum / times, ssim_base_sum / times, 
                      mse_sum / times, mse_base_sum / times, mae_sum / times, mae_base_sum / times, mape_sum / times, 
                      mape_base_sum / times, rmse_sum / times, rmse_base_sum / times)

            if self.log_level <= logging.INFO:
                img_input = inputs[-1].cpu().detach() 
                img_label = labels[-1].cpu().detach() 
                img_mask = masks[-1].cpu().detach() 
                img_output = output[-1].cpu().detach().numpy()

                self.log.info('Training Image')
                Plotter.plot_images(img_input, img_label, img_mask, img_output)

            # Validation
            if self.log_level <= logging.WARNING:
                self.log.info('-'*50)
                self.log.info('Validation')
                self.log.info('-'*50)
                self.log.info('Validate Epoch: %d/%d', epoch, epochs)

            model.eval()
            batches_trained = 0
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
                for times, (inputs, labels, masks) in enumerate(val_loader, 1):
                    (inputs, labels, masks) = (inputs.to(device), labels.to(device), masks.to(device))
                    val_len = len(val_loader.dataset) * inputs.shape[-1]

                    for i in range(inputs.shape[-1]):
                        batches_trained += len(inputs)

                        inputs_s = inputs[:,:,:,:,i].reshape(len(inputs), 3, 64, 64)
                        masks_s = masks[:,:,:,:,i].reshape(len(inputs), 1, 64, 64)
                        labels_s = labels[:,:,:,:,i].reshape(len(inputs), 1, 64, 64)

                        output = model(inputs_s)
                        output *= masks_s
                        
                        mae = mae_loss(output, labels_s)
                        mae_sum += mae.item()

                        baseline = masks_s * 0.5
                        mae_base = mae_loss(baseline, labels_s)
                        mae_base_sum += mae_base.item()

                        ssim = ssim_loss(output, labels_s)
                        ssim_sum += ssim
                        ssim_base = ssim_loss(baseline, labels_s)
                        ssim_base_sum += ssim_base
                        mse = mse_loss(output, labels_s)
                        mse_sum += mse
                        mse_base = mse_loss(baseline, labels_s)
                        mse_base_sum += mse_base
                        mape = mape_loss(output, labels_s)
                        mape_sum += mape
                        mape_base = mape_loss(baseline, labels_s)
                        mape_base_sum += mape_base
                        rmse = rmse_loss(output, labels_s)
                        rmse_sum += rmse
                        rmse_base = rmse_loss(baseline, labels_s)
                        rmse_base_sum += rmse_base

                        # Show progress
                        self.log.debug('Validate Epoch: %d/%d [%d/%d (%.0f%%)] Image: %d\nSSIM: %.3f/%.3f\tMSE: %.3f/%.3f\tMAE: %.3f/%.3f\tMAPE: %.3f/%.3f\tRMSE: %.3f/%.3f\n',
                                  epoch, epochs, batches_trained, val_len, 100. * batches_trained / val_len, i, ssim, ssim_base, mse, mse_base, mae, mae_base, mape, mape_base, rmse, rmse_base)

                times *= inputs.shape[-1]
                val_loss_mse = np.append(val_loss_mse, mse_sum.cpu() / times)
                val_loss_mae = np.append(val_loss_mae, mae_sum / times)
                self.log.debug('Average SSIM: %.3f/%.3f\tAverage MSE: %.3f/%.3f\tAverage MAE: %.3f/%.3f\tAverage MAPE: %.3f/%.3f\tAverage RMSE: %.3f/%.3f', 
                          ssim_sum / times, ssim_base_sum / times, 
                          mse_sum / times, mse_base_sum / times, mae_sum / times, mae_base_sum / times, mape_sum / times, 
                          mape_base_sum / times, rmse_sum / times, rmse_base_sum / times)

                if self.log_level <= logging.INFO:
                    img_label = labels[-1].cpu().detach() 
                    img_input = inputs[-1].cpu().detach() 
                    img_mask = masks[-1].cpu().detach() 
                    img_output = output[-1].cpu().detach().numpy()

                    self.log.info('Validation Image')
                    Plotter.plot_images(img_input, img_label, img_mask, img_output)

            if optimize:
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                checkpoint = Checkpoint.from_dict(checkpoint_data)

                session.report(
                    {'mae': mae_sum / times, 'mse': mse_sum / times, 'mape': mape_sum / times, 'rmse': rmse_sum / times, 'ssim': ssim_sum / times},
                    checkpoint=checkpoint,
                )

            torch.cuda.empty_cache()

            # Early stopping
            current_loss = mae_sum / times
            self.log.debug('The Current Loss: %.3f', current_loss)

            if current_loss >= last_loss:
                trigger_times += 1
                self.log.debug('Trigger Times: %d', trigger_times)

                if trigger_times >= patience:
                    self.log.debug('Early stopping!\nStart to test process.')
                    return

            else:
                self.log.debug('trigger times: 0')
                trigger_times = 0

            last_loss = current_loss
            
            train_losses = {
                'MSE': train_loss_mse,
                'MAE': train_loss_mae
            }
            val_losses = {
                'MSE': val_loss_mse,
                'MAE': val_loss_mae
            }
            Plotter.plot_loss(train_losses, val_losses)

        return model
    
    def optimize_unet(self, config, num_trials=10, cpus_per_trial=1, gpus_per_trial=1):
        scheduler = ASHAScheduler(
            metric='mae',
            mode='min',
            max_t=config['epochs'],
            grace_period=1,
            reduction_factor=2,
        )

        result = tune.run(
            partial(self.train_unet, optimize=True),
            resources_per_trial={'cpu': cpus_per_trial, 'gpu': gpus_per_trial},
            config=config,
            num_samples=num_trials,
            scheduler=scheduler,
            local_dir='./ray_results'
        )

        best_trial = result.get_best_trial('mae', 'min')
        print('-'*50)
        print('Best trial config: {}'.format(best_trial.config))
        print('Best trial final validation mae: {}'.format(best_trial.last_result['mae']))
        print('Best trial final validation mse: {}'.format(best_trial.last_result['mse']))
        print('Best trial final validation mape: {}'.format(best_trial.last_result['mape']))
        print('Best trial final validation rmse: {}'.format(best_trial.last_result['rmse']))
        print('Best trial final validation ssim: {}'.format(best_trial.last_result['ssim']))
        print('-'*50)

        return best_trial
        
    