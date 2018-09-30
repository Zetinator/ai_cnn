import numpy as np
import glob
import os
import cv2

image_paths = sorted(glob.glob('*.png'), key=os.path.getmtime)
images = [cv2.imread(img,0) for img in image_paths]
images = np.array(images)
images = images.reshape(np.append(images.shape,1))
images = images/images.max()

np.savez('compressed_data.npz', images = images)
