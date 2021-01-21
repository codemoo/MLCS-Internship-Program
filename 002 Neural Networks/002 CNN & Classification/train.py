#==========================================#
# Title:  Data Loader
# Author: Hwanmoo Yong
# Date:   2021-01-17
#==========================================#
from data_loader import DataLoader
from cnn_network import RoadClassificationModel
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau, Callback, TensorBoard

import sys
import numpy as np

import os
if not os.path.exists('./models'):
    os.makedirs('./models')

data_type = "CNN"
batch_size = 4048

def split(flag, d):
    train_idx = int(d.shape[0]*0.7)
    eval_idx = int(d.shape[0]*0.9)

    if flag == "train":
        return d[0:train_idx]
    elif flag == "val":
        return d[train_idx:eval_idx]
    elif flag == "test":
        return d[eval_idx:]

def main(window_size):
    dl = DataLoader()
    
    # Create NPZ from csv raw dataset
    # dl.create()

    # Load and Create windowed NPZ
    # d = dl.read(window_size=int(window_size))

    # Load windowed dataset
    d = dl.load(window_size=int(window_size))

    rm = RoadClassificationModel(time_window=int(window_size))
    model = rm.build()

    x_train = [
                split("train",d['ba']).reshape(-1,int(window_size),1),
                split("train",d['u']),
                split("train",d['K_seq'])
            ]
    y_train = [split("train",d['a']),split("train",d['K'])]

    x_val = [
                split("val",d['ba']).reshape(-1,int(window_size),1),
                split("val",d['tire_stft']).reshape(-1,25,21,1),
                split("val",d['K_seq'])
            ]
    y_val = [split("val",d['a']),split("val",d['K'])]

    if not os.path.exists('./models/'+sys.argv[2] + '_'  +  data_type):
        os.makedirs('./models/'+sys.argv[2] + '_'  +  data_type)

    earlyStopping = EarlyStopping(monitor='val_loss', patience=21, verbose=0, mode='min')
    checkpointer = ModelCheckpoint(monitor='val_loss', filepath='./models/'+sys.argv[2] + '_' + data_type+'/{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
    rl_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    csv_logger = CSVLogger('./models/'+sys.argv[2] + '_'  +  data_type+'/result.csv')

    model.fit(x_train, y_train,
            batch_size=batch_size, epochs=10000,
            validation_data=(x_val, y_val),
            shuffle=True,
            callbacks=[earlyStopping, checkpointer, rl_scheduler, csv_logger])

if __name__=="__main__":
    main(sys.argv[1])