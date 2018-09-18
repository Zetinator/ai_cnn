import numpy as np
import cv2
import os, glob

# Utility class
class Settings(object):
    def __init__(self):
        self.ydim = 160
        self.xdim = 90
        self.channels = 1
        self.model_weights = None

class DataLoader(object):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.images = None
        self.outputs = None

    def load_outputs (self):
        self.outputs = genfromtxt(os.path.join(self.dataset_path, '*outputs.txt'), delimiter=',')

    def get_outputs (self):
        return self.normalize(self.outputs)

    def load_images(self):
        image_paths = sorted(glob.glob(os.path.join(self.dataset_path, '*.png')))
        images = [cv2.imread(img,0) for img in image_paths]
        self.images = np.array(images)

    def get_training (self):
        training = self.images[:int(np.ceil(self.images.shape[0]*.7)),:,:]

        return self.normalize(training)

    def get_test (self):
        test = self.images[int(np.ceil(self.images.shape[0]*.7)):,:,:]

        return self.normalize(test)

    def get_all (self):
        test = self.normalize(self.images[:,:,:])

        return self.nomalize(test)

    def nomalize (self, data):
        # normalize data between 0 and 1
        data = (data/data.max()).astype('uint8')

        return data

