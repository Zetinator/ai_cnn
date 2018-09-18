from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras.layers import Dense, Dropout
from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.layers import ZeroPadding2D, BatchNormalization, Activation
from keras.layers import UpSampling2D
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.models import load_model, Model
from keras.layers.pooling import MaxPooling2D
from keras.utils import plot_model

import numpy as np
# from utils import Settings


class CNN(object):
    def __init__(self, settings):
        self.settings = settings
        self.xdim = self.settings.xdim
        self.ydim = self.settings.ydim
        self.channels = self.settings.channels
        self.model = None

    def build_model(self, lr=1e-3):
        dropout = 0.2
        shape=(None, self.ydim, self.xdim, self.channels)
        input = Input(batch_shape=shape)


        # reference
        xin = Conv2D(filters=16, kernel_size=5, padding='same')(input)


        # skip layer tensor
        skip = Conv2D(filters=1,
                       kernel_size=5,
                       padding='same',
                       dilation_rate=2)(input)


        # image reduced by 8
        x8 = MaxPooling2D(8)(xin)
        x8 = BatchNormalization()(x8)
        x8 = Activation('relu', name='contract')(x8)


        # parallel stage
        dilation_rate = 1
        y = x8
        for i in range(4):
            a = Conv2D(filters=32,
                       kernel_size=5,
                       padding='same',
                       dilation_rate=dilation_rate)(x8)
            a = Dropout(dropout)(a)
            y = keras.layers.concatenate([a, y])
            dilation_rate += 1


        # disparity network
        # dense interconnection inspired by DenseNet
        dilation_rate = 1
        x = MaxPooling2D(8)(xin)
        for i in range(4):
            x = keras.layers.concatenate([x, y])
            y = BatchNormalization()(x)
            y = Activation('relu')(y)
            y = Conv2D(filters=64,
                       kernel_size=1,
                       padding='same')(y)
            y = BatchNormalization()(y)
            y = Activation('relu')(y)
            y = Conv2D(filters=16,
                       kernel_size=5,
                       padding='same',
                       dilation_rate=dilation_rate)(y)
            y = Dropout(dropout)(y)
            dilation_rate += 1


        # input image skip connection to disparity estimate
        x = keras.layers.concatenate([x, skip])
        y = BatchNormalization()(x)
        y = Activation('relu')(y)
        y = Conv2D(filters=16, kernel_size=5, padding='same')(y)

        
        # output
        output = Dense(9, activation='linear',
                     use_bias=True,
                     bias_initializer='zeros')(y)
        # CNN model
        self.model = Model(input,output)
        if self.settings.model_weights:
            print("Loading checkpoint weights %s...."
                  % self.settings.model_weights)
            self.model.load_weights(self.settings.model_weights)
        # self.model.compile(loss='mse',
                           # optimizer=RMSprop(lr=lr))
        self.model.compile(loss='binary_crossentropy',
                           optimizer=RMSprop(lr=lr))
        print("CNN Model:")
        self.model.summary()
        plot_model(self.model, to_file='densemapnet.png', show_shapes=True)
        return self.model
