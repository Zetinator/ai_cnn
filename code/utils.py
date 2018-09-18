import numpy as np
import cv2
import os, glob

# Utility class
class Settings(object):
    def __init__(self):
        self.ydim = 960
        self.xdim = 540
        self.channels = 3
        self.model_weights = None

class data_loader(object):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
    def load_images(self):
        img = np.aray([])
        for infile in sorted(glob.glob(os.path.join(self.dataset_path, '*.png'))):
            img np.append(img, cv2.imread(infile,0),axis=1)
