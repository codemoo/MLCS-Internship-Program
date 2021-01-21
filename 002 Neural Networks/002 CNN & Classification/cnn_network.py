#==========================================#
# Title:  Data Loader
# Author: Hwanmoo Yong
# Date:   2021-01-17
#==========================================#
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Dense, Input, Dropout, Concatenate, BatchNormalization, Activation, Bidirectional, GaussianNoise, Flatten
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau, Callback, TensorBoard
from keras.regularizers import l2
import keras.backend.tensorflow_backend as K
from keras.utils import multi_gpu_model
import keras

import numpy as np
import time

import tensorflow as tf

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

import sys

class RoadClassificationModel():
    def __init__(self, time_window=8):
        self.time_window = time_window

        # self.build()

    def build(self, front_trainable=True):
        _tsi = Input(shape=(25,21,2), name="tire_stft")

        # base_model = keras.applications.densenet.DenseNet121(include_top=False, weights=None, input_tensor=_tsi, input_shape=None, pooling=None, classes=1000)
        base_model = keras.applications.ResNet50(include_top=False, weights=None, input_tensor=_tsi, input_shape=None, pooling=None, classes=1000).output
        base_model = Flatten()(base_model)
        def fc(num_classes, _input, names, name, activation, trainable):
            x = _input
        
            x = Dense(128, kernel_initializer='glorot_normal',
                            kernel_regularizer=l2(0.001), trainable=trainable, name=names+'_d2_'+name)(x)
            x = BatchNormalization(trainable=trainable, name=names+'_b2_'+name)(x)
            x = Activation('elu', trainable=trainable, name=names+'_a2_'+name)(x)
            x = Dropout(0.5)(x)

            x = Dense(64, kernel_initializer='glorot_normal',
                            kernel_regularizer=l2(0.001), trainable=trainable, name=names+'_d3_'+name)(x)
            x = BatchNormalization(trainable=trainable, name=names+'_b3_'+name)(x)
            x = Activation('elu', trainable=trainable, name=names+'_a3_'+name)(x)
            x = Dropout(0.5)(x)
            return Dense(num_classes, activation=activation, name=name+'_out', trainable=trainable)(x)

        a_prediction = fc(1, base_model, "a_layer", "a", "linear", True)
        K_prediction = fc(1, base_model, "K_layer", "K", "linear", True)

        model = Model(inputs=[_tsi], outputs=[a_prediction, K_prediction])

        model.summary()

        model = multi_gpu_model(model, gpus=4)
        model.compile(  loss="MSE",
                        optimizer='adam',
                        metrics=['mse'])

        return model