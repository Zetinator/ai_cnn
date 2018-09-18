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
        if settings.model_weights:
            self.predict()
            return
        else:
            self.train_all()
            # epochs = 1
            # decay = 1e-6
            # lr = 1e-3 + decay
            # for i in range(500):
                # lr = lr - decay
                # print("Epoch: ", (i+1), " Learning rate: ", lr)
                # self.train_batch(epochs=epochs, lr=lr, seq=(i+1))
                # self.predict()

    def train_all(self, epochs=1000, lr=1e-3):
        checkdir = "checkpoint"
        try:
            os.mkdir(checkdir)
        except FileExistsError:
            print("There is already a 'checkpoint' folder... ")


        # callbacks
        filename = self.settings.dataset
        filename += ".densemapnet.weights.{epoch:02d}.h5"
        filepath = os.path.join(checkdir, filename)
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     save_weights_only=True,
                                     verbose=1,
                                     save_best_only=False)
        tb = TensorBoard(log_dir='./graph', histogram_freq=0,
                            write_graph=True, write_images=True)
        callbacks = [checkpoint, tb]

        if self.network is None:
            self.network = CNN(settings=self.settings)
            self.model = self.network.build_model(lr=lr)

        else:
            print("Using loss=crossent on linear output layer")
            self.model.compile(loss='binary_crossentropy',
                               optimizer=RMSprop(lr=lr, decay=1e-6))

        x = [self.train_lx, self.train_rx]
        self.model.fit(self.image_data,
                       self.outputs_data,
                       epochs=epochs,
                       batch_size=4,
                       shuffle=True,
                       callbacks=callbacks)

    def mkdir_outputs(self):
        dirname = 'outputs'

        filepath = os.path.join(self.dataset, dirname)
        os.makedirs(filepath, exist_ok=True)

    def predict(self, image):
        input_image = image

        print("Saving images on folder...")
        start_time = time.time()
        prediction = self.model.predict(input_image)
        elapsed_time += (time.time() - start_time)
        print("Prediction --> ", str(prediction))
        print("                             Elapsed time --> ", elapsed_time)

        return prediction


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load checkpoint hdf5 file of model trained weights"
    parser.add_argument("-w",
                        "--weights",
                        help=help_)
    parser.add_argument("-d",
                        "--dataset",
                        help="Name of dataset to load")
    help_ = "No training. Just prediction based on test data. Must load weights!"
    parser.add_argument("-p",
                        "--predict",
                        action='store_true',
                        help=help_)

    args = parser.parse_args()
    settings = Settings()
    settings.model_weights = args.weights
    settings.dataset = args.dataset
    settings.predict = args.predict

    predictor = Predictor(settings=settings)
    if settings.predict:
        predictor.predict()
    else:
        predictor.train_network()
