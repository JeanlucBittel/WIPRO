import numpy as np
import torch

from torch.utils.data import Dataset

class NIfTIDataset(Dataset):
    
    def __init__(self, images, transforms, image_size):
        self.input_contrats = images[0]
        self.flair_contrasts = images[1]
        self.masks = images[2]
        self.transforms = transforms[0]
        self.transforms_flair = transforms[1]
        self.transforms_mask = transforms[2]
        self.image_size_x = image_size[0]
        self.image_size_y = image_size[1]

    def __len__(self):
        return len(self.input_contrats)

    def __getitem__(self, index):
        # Load Images
        t1 = np.load(self.input_contrats[index][0])[:,:,0:2]
        t1gd = np.load(self.input_contrats[index][1])[:,:,0:2]
        t2 = np.load(self.input_contrats[index][2])[:,:,0:2]
        flair = np.load(self.flair_contrasts[index])[:,:,0:2]
        mask = np.load(self.masks[index])[:,:,0:2]
        
        dim_contrast = (3, self.image_size_x, self.image_size_y, 2)
        dim_flair = (1, self.image_size_x, self.image_size_y, 2)

        # Concatenate images            
        concat_constrasts = np.zeros(dim_contrast, dtype=np.float32)
        concat_constrasts[0] = t1.astype(np.float32)
        concat_constrasts[1] = t1gd.astype(np.float32)
        concat_constrasts[2] = t2.astype(np.float32)
        concat_constrasts = torch.from_numpy(concat_constrasts)

        flair_channel = np.zeros(dim_flair, dtype=np.float32)
        flair_channel[0] = flair.astype(np.float32)
        flair_channel = torch.from_numpy(flair_channel)

        mask_channel = np.zeros(dim_flair, dtype=np.float32)
        mask_channel[0] = mask.astype(np.float32)
        mask_channel = torch.from_numpy(mask_channel)
        
        train = self.transforms(concat_constrasts)
        flair = self.transforms_flair(flair_channel)
        mask = self.transforms_mask(mask_channel)
        
        return train, flair, mask 