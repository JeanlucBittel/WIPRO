import os
import numpy as np

from tqdm import tqdm

class CustomDataset():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
    def load_images(self):
        # All Images
        image_files = []
        image_flairs = []
        image_masks = []
        patients = []

        print('Load Images: ')
        for patient in tqdm(os.listdir(self.data_dir)):
            # Get Image paths
            patients.append(patient)
            patient_path = os.path.join(self.data_dir, patient)
            t1_path = os.path.join(patient_path, patient + '_T1_resized.npy')
            t1gd_path = os.path.join(patient_path, patient + '_T1GD_resized.npy')
            t2_path = os.path.join(patient_path, patient + '_T2_resized.npy')
            flair_path = os.path.join(patient_path, patient + '_FLAIR_resized.npy')
            mask_path = os.path.join(patient_path, patient + '_MASK_resized.npy')

            # Concatenate images
            image_files.append([t1_path, t1gd_path, t2_path])
            image_flairs.append(flair_path)
            image_masks.append(mask_path)
            
        self.image_files = np.array(image_files)
        self.image_flairs = np.array(image_flairs)
        self.image_masks = np.array(image_masks)
    
    def train_val_test_split(self):
        val_frac = 0.15
        test_frac = 0.15
        num_total = len(self.image_files)
        indices = np.arange(num_total)
        np.random.shuffle(indices)

        test_split = int(test_frac * num_total)
        val_split = int(val_frac * num_total) + test_split
        test_indices = indices[:test_split]
        val_indices = indices[test_split:val_split]
        train_indices = indices[val_split:]

        train_x = [self.image_files[i] for i in train_indices]
        train_y = [self.image_flairs[i] for i in train_indices]
        train_mask = [self.image_masks[i] for i in train_indices]
        val_x = [self.image_files[i] for i in val_indices]
        val_y = [self.image_flairs[i] for i in val_indices]
        val_mask = [self.image_masks[i] for i in val_indices]
        test_x = [self.image_files[i] for i in test_indices]
        test_y = [self.image_flairs[i] for i in test_indices]
        test_mask = [self.image_masks[i] for i in test_indices]
        
        return [train_x, train_y, train_mask], [val_x, val_y, val_mask], [test_x, test_y, test_mask]