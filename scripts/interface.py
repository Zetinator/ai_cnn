from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras.callbacks import ModelCheckpoint, LambdaCallback, TensorBoard
from keras.optimizers import RMSprop, SGD

import numpy as np

import argparse
import os
from os import path
import time
import matplotlib.image as img
import matplotlib.pyplot as plt
from scipy import misc

from utils import Settings
from utils import DataLoader
from model_dense import CNN


class Predictor(object):
    def __init__(self, settings=Settings()):
        # some variables
        self.settings = settings
        self.dataset = settings.dataset
        self.image_data = None
        self.outputs_data = None
        self.network = None

        # action time
        self.mkdir_outputs()
        self.load_data()
        self.get_max_disparity()

    def load_data():
        loader = DataLoader(self.dataset)
        print('Loading data... ')
        loader.load_images()
        self.image_data = loader.get_all()
        loader.load_outputs()
        self.outputs_data = loader.get_outputs()
        print('                     --> SUCCESS')

    def train_network(self):
        if self.settings.num_dataset == 1:
            self.train_all()
            return

        if self.settings.notrain:
            self.train_batch()
            return

        epochs = 1
        if self.settings.otanh:
            decay = 2e-6
            lr = 1e-2 + decay
        else:
            decay = 1e-6
            lr = 1e-3 + decay
        for i in range(400):
            lr = lr - decay
            print("Epoch: ", (i+1), " Learning rate: ", lr)
            self.train_batch(epochs=epochs, lr=lr, seq=(i+1))
            self.predict_disparity()

    def train_all(self, epochs=1000, lr=1e-3):
        checkdir = "checkpoint"
        try:
            os.mkdir(checkdir)
        except FileExistsError:
            print("Folder exists: ", checkdir)

        filename = self.settings.dataset
        filename += ".densemapnet.weights.{epoch:02d}.h5"
        filepath = os.path.join(checkdir, filename)
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     save_weights_only=True,
                                     verbose=1,
                                     save_best_only=False)
        predict_callback = LambdaCallback(on_epoch_end=lambda epoch,
                                          logs: self.predict_disparity())
        tb = TensorBoard(log_dir='./Graph', histogram_freq=0,
          write_graph=True, write_images=True)
        callbacks = [checkpoint, predict_callback, tb]
        self.load_train_data(1)
        if self.network is None:
            self.network = DenseMapNet(settings=self.settings)
            self.model = self.network.build_model(lr=lr)

        if self.settings.otanh:
            print("Using loss=mse on tanh output layer")
            self.model.compile(loss='mse',
                               optimizer=RMSprop(lr=lr, decay=1e-6))
        else:
            print("Using loss=crossent on sigmoid output layer")
            self.model.compile(loss='binary_crossentropy',
                               optimizer=RMSprop(lr=lr, decay=1e-6))

        if self.settings.model_weights:
            if self.settings.notrain:
                self.predict_disparity()
                return

        x = [self.train_lx, self.train_rx]
        self.model.fit(x,
                       self.train_dx,
                       epochs=epochs,
                       batch_size=4,
                       shuffle=True,
                       callbacks=callbacks)

    def mkdir_outputs(self):
        dirname = 'outputs'

        filepath = os.path.join(self.dataset, dirname)
        os.makedirs(filepath, exist_ok=True)


    def get_epe(self, use_train_data=True, get_performance=False):
        if use_train_data:
            lx = self.train_lx
            rx = self.train_rx
            dx = self.train_dx
            if self.settings.mask:
                print("Using mask images")
                mx = self.train_mx
            print("Using train data... Size: ", lx.shape[0])
        else:
            lx = self.test_lx
            rx = self.test_rx
            dx = self.test_dx
            if self.settings.mask:
                print("Using mask images")
                mx = self.test_mx
            if self.settings.predict:
                print("Using complete data... Size: ", lx.shape[0])
            else:
                print("Using test data... Size: ", lx.shape[0])

        # sum of all errors (normalized)
        epe_total = 0
        # count of images
        t = 0
        nsamples = lx.shape[0]
        elapsed_total = 0.0
        if self.settings.images:
            print("Saving images on folder...")
        for i in range(0, nsamples, 1):
            indexes = np.arange(i, i + 1)
            left_images = lx[indexes, :, :, : ]
            right_images = rx[indexes, :, :, : ]
            disparity_images = dx[indexes, :, :, : ]
            if self.settings.mask:
                mask_images = mx[indexes, :, :, : ]
            # measure the speed of prediction on the 10th sample to avoid variance
            if get_performance:
                start_time = time.time()
                predicted_disparity = self.model.predict([left_images, right_images])
                elapsed_total += (time.time() - start_time)
            else:
                predicted_disparity = self.model.predict([left_images, right_images])

            predicted = predicted_disparity[0, :, :, :]
            # if self.settings.mask:
            # ground_mask = np.ceil(mask_images[0, :, :, :])
            # predicted = np.multiply(predicted, ground_mask)

            ground = disparity_images[0, :, :, :]
            if self.settings.mask:
                ground_mask = mask_images[0, :, :, :]
                dim = np.count_nonzero(ground_mask)
                nz = np.nonzero(ground_mask)
                epe = predicted[nz] - ground[nz]
            else:
                dim = predicted.shape[0] * predicted.shape[1]
                epe = predicted - ground
            # normalized error on all pixels
            epe = np.sum(np.absolute(epe))
            # epe = epe.astype('float32')
            epe = epe / dim
            epe_total += epe

            if (get_performance and self.settings.images) or ((i%100) == 0):
                path = "test"
                if use_train_data:
                    path = "train"
                filepath  = os.path.join(self.images_pdir, path)
                left = os.path.join(filepath, "left")
                right = os.path.join(filepath, "right")
                disparity = os.path.join(filepath, "disparity")
                prediction = os.path.join(filepath, "prediction")
                filename = "%04d.png" % i
                left = os.path.join(left, filename)
                if left_images[0].shape[2] == 1:
                    self.predict_images(left_images[0], left)
                else:
                    plt.imsave(left, left_images[0])

                right = os.path.join(right, filename)
                if right_images[0].shape[2] == 1:
                    self.predict_images(right_images[0], right)
                else:
                    plt.imsave(right, right_images[0])
                self.predict_images(predicted, os.path.join(prediction, filename))
                self.predict_images(ground, os.path.join(disparity, filename))

        epe = epe_total / nsamples
        # epe in pix units
        epe = epe * self.dmax
        if self.settings.dataset == "kitti2015":
            epe = epe / 256.0
            print("KITTI 2015 EPE: ", epe)
        else:
            print("EPE: %0.2fpix" % epe)
        if epe < self.best_epe:
            self.best_epe = epe
            print("------------------- BEST EPE : %f ---------------------" % epe)
            tmpdir = "tmp"
            try:
                os.mkdir(tmpdir)
            except FileExistsError:
                print("Folder exists: ", tmpdir)
            filename = open('tmp/epe.txt', 'a')
            datetime = time.strftime("%H:%M:%S")
            filename.write("%s : %s EPE: %f\n" % (datetime, self.settings.dataset, epe))
            filename.close()
        # speed in sec
        if get_performance:
            print("Speed: %0.4fsec" % (elapsed_total / nsamples))
            print("Speed: %0.4fHz" % (nsamples / elapsed_total))

    def predict_images(self, image, filepath):
        size = [image.shape[0], image.shape[1]]
        if self.settings.otanh:
            image += 1.0
            image =  np.clip(image, 0.0, 2.0)
            image *= (255*0.5)
        else:
            image =  np.clip(image, 0.0, 1.0)
            image *= 255

        image = image.astype(np.uint8)
        image = np.reshape(image, size)
        misc.imsave(filepath, image)

    def predict_disparity(self):
        if self.settings.predict:
            if self.network is None:
                self.network = DenseMapNet(settings=self.settings)
                self.model = self.network.build_model()
            # gpu is slow in prediction during initial load of data
            # distorting the true speed of the network
            # we get the speed after 1 prediction
            if self.settings.images:
                self.get_epe(use_train_data=False, get_performance=True)
            else:
                for i in range(4):
                    self.get_epe(use_train_data=False, get_performance=True)
        else:
            # self.settings.images = True
            self.get_epe(use_train_data=False, get_performance=True)
            # self.get_epe(use_train_data=False)
            if self.settings.notrain:
                self.get_epe()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load checkpoint hdf5 file of model trained weights"
    parser.add_argument("-w",
                        "--weights",
                        help=help_)
    parser.add_argument("-d",
                        "--dataset",
                        help="Name of dataset to load")
    help_ = "No training. Just prediction based on test data. Must load weights."
    parser.add_argument("-p",
                        "--predict",
                        action='store_true',
                        help=help_)
    help_ = "Generate outputs during prediction. Outputs are stored outputs/"
    parser.add_argument("-t",
                        "--test",
                        action='store_true',
                        help=help_)


    args = parser.parse_args()
    settings = Settings()
    settings.model_weights = args.weights
    settings.dataset = args.dataset
    settings.predict = args.predict
    settings.test = args.test

    predictor = Predictor(settings=settings)
    if settings.predict:
        predictor.predict_disparity()
    else:
        predictor.train_network()
