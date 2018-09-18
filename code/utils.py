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

class data_loader(object):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.images = None

    def load_images(self):
        image_paths = sorted(glob.glob(os.path.join(self.dataset_path, '*.png')))
        images = [cv2.imread(img,0) for img in image_paths]
        self.images = np.array(images)

    def get_training (self):
        training = self.images[:int(np.ceil(self.images.shape[0]*.7)),:,:]

        return training

    def get_test (self):
        test = self.images[int(np.ceil(self.images.shape[0]*.7)):,:,:]

        return test
