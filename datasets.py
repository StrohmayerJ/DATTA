from torch.utils.data import Dataset
import numpy as np
import albumentations as A
import yaml
import os
import pickle

# suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 52 L-LTF subcarriers 
csi_vaid_subcarrier_index = []
csi_vaid_subcarrier_index += [i for i in range(6, 32)]
csi_vaid_subcarrier_index += [i for i in range(33, 59)]
# 56 HT-LTF: subcarriers 
#csi_vaid_subcarrier_index += [i for i in range(66, 94)]     
#csi_vaid_subcarrier_index += [i for i in range(95, 123)] 
CSI_SUBCARRIERS = len(csi_vaid_subcarrier_index) 

########################################################################
#################### Widar3.0-G6 DOMAIN Dataset ########################
########################################################################

class Widar3g6d(Dataset):
    def __init__(self, dataPath, augPath,opt,mode):
        self.dataPath = dataPath
        folderEndIndex = dataPath.rfind('/')
        self.dataFolder = self.dataPath[0:folderEndIndex]
        self.params = None
        self.augPath = augPath
        self.opt = opt
        self.mode = mode
        self.data_cache = self.dataFolder+'/widar3-g6_csi_domain_train_cache.pkl' if self.mode == 'TRAIN' else self.dataFolder+'/widar3-g6_csi_domain_test_cache.pkl'
        
        # read augmentation parameters from yaml file
        if self.augPath != '':
            with open(self.augPath, 'r') as file:
                self.params = yaml.safe_load(file)

        # load complex CSI cache to speed up training or quit if cache not found (see data/widar3g6d/data.md)
        if os.path.exists(self.data_cache):
            print("Loading Widar3.0-G6D CSI cache...")
            with open(self.data_cache, 'rb') as f:
                loaded_data = pickle.load(f)
            self.csiComplex = loaded_data['csiComplex']
            self.c = loaded_data['activities']
            self.e = loaded_data['environments']
            self.u = loaded_data['users']
            self.d = loaded_data['domains']
            self.T_MAX = loaded_data['T_MAX']
        else:
            print("Widar3.0-G6D CSI cache not found! quitting... (see data/widar3g6d/data.md)")
            quit()
    
        # compute base features from complex CSI
        self.features = np.abs(self.csiComplex)

        # number of samples 
        self.dataSize = len(self.features)  

    def __len__(self):
        return self.dataSize
    
    def augmentSample(self, featureWindow):
        if np.random.rand() < self.params['amplitudeProbability']:
            featureWindow = A.RandomBrightnessContrast(p=1, brightness_limit=(-self.params['amplitudeRange'], self.params['amplitudeRange']), contrast_limit=0, always_apply=True)(image=featureWindow)['image']
        if np.random.rand() < self.params['pixelWiseDropoutProbability']:
            featureWindow = pixelwiseDropout(self,featureWindow)
            featureWindow = featureWindow.astype(np.float32)
        if np.random.rand() < self.params['columnWiseDropoutProbability']:
            featureWindow = columnwiseDropout(self,featureWindow)
            featureWindow = featureWindow.astype(np.float32)
        if np.random.rand() < self.params['circRotationProbability']:
            featureWindow = A.ShiftScaleRotate(shift_limit_x=self.params['circRotationRange'], shift_limit_y=0, rotate_limit=0,border_mode=4,always_apply=True)(image=featureWindow)['image']
        if np.random.rand() < self.params['hflipProbability']:
            featureWindow = A.HorizontalFlip(always_apply=True)(image=featureWindow)['image']
        return featureWindow

    def __getitem__(self, index):
        c = self.c[index]-1 # activity class labels: [1,2,3,4,5,6] 
        e = self.e[index] # environment ids: [1,2,3]
        u = self.u[index] # user ids: [1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17] 
        d = self.d[index]-9 if self.mode == 'TRAIN' else self.d[index] # [9,10,11,12,13,14,15] are TRAIN domain labels, [0,1,2,3,4,5,6,7,8] are TEST domain labels
        
        # extract feature window
        featureWindow = self.features[index]
        featureWindow = np.expand_dims(featureWindow, axis=0)

        # apply augmentations
        if self.augPath != '':
            featureWindow = self.augmentSample(featureWindow)

        return featureWindow, c, d, e, u # [featureWindow, activity, domain, environment, user]


# applies pixel-wise dropout augmentation
def pixelwiseDropout(self, featureWindow):
    mask = np.random.choice([0, 1], size=featureWindow.shape, p=[self.params['pixelWiseDropout'], 1 - self.params['pixelWiseDropout']])
    img_copy = featureWindow.copy()
    img_mean = np.mean(featureWindow)
    if np.random.rand() < self.params['pixelWiseDropoutZeroPixels']:
        img_copy *= mask  # replace with 0
    else:
        img_copy = img_copy * mask + img_mean * (1 - mask)  # replace with CSI mean
    return img_copy

# applies column-wise dropout augmentation
def columnwiseDropout(self, featureWindow):
    mask = np.random.choice([0, 1], size=featureWindow.shape[1:], p=[self.params['columnWiseDropout'], 1 - self.params['columnWiseDropout']])
    img_copy = featureWindow.copy()
    img_mean = np.mean(featureWindow)
    if np.random.rand() < self.params['columnWiseDropoutZeroPixels']:
        img_copy *= mask[None, :]  # replace with 0
    else:
        img_copy = img_copy * mask[None, :] + img_mean * (1 - mask[None, :])  # replace with CSI mean
    return img_copy


