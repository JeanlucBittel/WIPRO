import os
import nibabel as nib
import skimage.transform as skTrans
import numpy as np

from tqdm import tqdm

class ImagePrep():
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
    def save_images(self, image_size):
        dim_start = 77
        dim_end = 79

        print('Save Images: ')
        for patient in tqdm(os.listdir(self.data_dir)):        
            # Get Image paths
            patient_path = os.path.join(self.data_dir, patient)
            t1_path = os.path.join(patient_path, patient + '_T1.nii.gz')
            t1gd_path = os.path.join(patient_path, patient + '_T1GD.nii.gz')
            t2_path = os.path.join(patient_path, patient + '_T2.nii.gz')
            flair_path = os.path.join(patient_path, patient + '_FLAIR.nii.gz')

            # Load Images
            t1 = nib.load(t1_path).get_fdata()[:,:,dim_start:dim_end]
            t1gd = nib.load(t1gd_path).get_fdata()[:,:,dim_start:dim_end]
            t2 = nib.load(t2_path).get_fdata()[:,:,dim_start:dim_end]
            flair = nib.load(flair_path).get_fdata()[:,:,dim_start:dim_end]

            # Resize images
            t1 = skTrans.resize(t1, image_size, order=1, preserve_range=True)
            t1gd = skTrans.resize(t1gd, image_size, order=1, preserve_range=True)
            t2 = skTrans.resize(t2, image_size, order=1, preserve_range=True)
            flair = skTrans.resize(flair, image_size, order=1, preserve_range=True)

            # Mask
            t1[t1<1] = 0
            t1gd[t1gd<1] = 0
            t2[t2<1] = 0
            flair[flair<1] = 0

            mask = (t1!=0) & (t1gd!=0) & (t2!=0)
            
            t1_norm, t1gd_norm, t2_norm, flair_norm = self.normalize(t1, t1gd, t2, flair, mask)

            # Remove Images
            if os.path.exists(os.path.join(patient_path, patient + '_T1_resized.npy')):
                os.remove(os.path.join(patient_path, patient + '_T1_resized.npy')) 
                os.remove(os.path.join(patient_path, patient + '_T1GD_resized.npy'))
                os.remove(os.path.join(patient_path, patient + '_T2_resized.npy'))
                os.remove(os.path.join(patient_path, patient + '_FLAIR_resized.npy'))
                os.remove(os.path.join(patient_path, patient + '_MASK_resized.npy'))

            # Save Images
            np.save(os.path.join(patient_path, patient + '_T1_resized'), t1_norm)
            np.save(os.path.join(patient_path, patient + '_T1GD_resized'), t1gd_norm)
            np.save(os.path.join(patient_path, patient + '_T2_resized'), t2_norm)
            np.save(os.path.join(patient_path, patient + '_FLAIR_resized'), flair_norm)
            np.save(os.path.join(patient_path, patient + '_MASK_resized'), mask)
    
    def normalize(self, t1, t1gd, t2, flair, mask):
        mask_flat = mask.copy().flatten()
        t1_flat = t1.copy().flatten()
        t1gd_flat = t1gd.copy().flatten()
        t2_flat = t2.copy().flatten()
        flair_flat = flair.copy().flatten()

        t1_mean = np.mean(t1_flat, where=mask_flat)
        t1_std = np.std(t1_flat, where=mask_flat)
        t1gd_mean = np.mean(t1gd_flat, where=mask_flat)
        t1gd_std = np.std(t1gd_flat, where=mask_flat)
        t2_mean = np.mean(t2_flat, where=mask_flat)
        t2_std = np.std(t2_flat, where=mask_flat)
        flair_mean = np.mean(flair_flat, where=mask_flat)
        flair_std = np.std(flair_flat, where=mask_flat)

        t1_norm = ((t1 - t1_mean) / (3 * t1_std) + 0.5) * mask
        t1gd_norm = ((t1gd - t1gd_mean) / (3 * t1gd_std) + 0.5) * mask
        t2_norm = ((t2 - t2_mean) / (3 * t2_std) + 0.5) * mask
        flair_norm = ((flair - flair_mean) / (3 * flair_std) + 0.5) * mask

        t1_norm[t1_norm<0] = 0
        t1_norm[t1_norm>1] = 1
        t1gd_norm[t1gd_norm<0] = 0
        t1gd_norm[t1gd_norm>1] = 1
        t2_norm[t2_norm<0] = 0
        t2_norm[t2_norm>1] = 1
        flair_norm[flair_norm<0] = 0
        flair_norm[flair_norm>1] = 1
        
        return t1_norm, t1gd_norm, t2_norm, flair_norm