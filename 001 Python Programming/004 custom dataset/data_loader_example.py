#==========================================#
# Title:  Data Loader
# Author: Hwanmoo Yong
# Date:   2021-01-08
#==========================================#
import os, glob
import csv
import numpy as np

import random

data_per_episode = 2000

class DataLoader():

    def __init__(self, dataset_name = 'K_test', test=False, shuffle=False, normalize=True):
        # Initialize variables
        self.body_acc = []
        self.tire_acc = []
        self.body_vel = []
        self.body_pos = []
        self.body_accf = []
        self.tire_accf = []
        self.tire_vel = []
        self.a = []
        self.G = []
        self.K = []
        self.u = []
        self.r = []
        self.def_vel = []

        # Settings
        self.normalize_flag = normalize
        self.dataset_name = dataset_name
        self.shuffle = shuffle

    def normalize(self, flag, arr):
        # Simple normalization using mean and max values.
        # Normally, we utilize mean and standard deviation.
        if self.normalize_flag:
            arr = arr - arr.mean()
            arr = arr / abs(arr).max()            

        return arr

    def create(self):
        # 'Create' imports all the raw dataset and convert the dataset into normalized numpy arrays.

        # This example code imports multiple csv files (sensory outputs).
        csvs = sorted(glob.glob(os.path.join('./datasets',self.dataset_name,'*.csv')))
        if self.shuffle:
            random.shuffle(csvs)
        
        for _csv in csvs:
            with open(_csv, newline='') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                next(csv_reader, None)  # skip the headers
                for row in csv_reader:
                    self.body_acc.append(   float(row[0]))
                    self.body_vel.append(   float(row[1]))
                    self.body_accf.append(  float(row[2]))
                    self.tire_acc.append(   float(row[3]))
                    self.tire_vel.append(   float(row[4]))
                    self.tire_accf.append(  float(row[5]))
                    self.a.append(float(row[6]))
                    self.G.append(float(row[7]))
                    self.K.append(float(row[8]))
                    self.u.append(float(row[9]))
                    self.r.append(float(row[10]))

        # Normalize the data and convert into numpy arrays
        self.body_acc   = self.normalize("body_acc", np.asarray(self.body_acc))
        self.tire_acc   = self.normalize("tire_acc", np.asarray(self.tire_acc))
        self.body_vel   = self.normalize("body_vel", np.asarray(self.body_vel))
        self.tire_vel   = self.normalize("tire_vel", np.asarray(self.tire_vel))
        self.def_vel    = self.normalize("dev_vel",  np.asarray(self.body_vel - self.tire_vel))
        self.body_accf  = self.normalize("body_accf", np.asarray(self.body_accf))
        self.tire_accf  = self.normalize("tire_accf", np.asarray(self.tire_accf))
        self.a          = self.normalize("a", np.asarray(self.a))
        self.G          = self.normalize("G", np.asarray(self.G))
        self.K = self.normalize("K", np.asarray(self.K))
        self.u = self.normalize("u", np.asarray(self.u))
        self.r = self.normalize("r", np.asarray(self.r))

        np.savez(os.path.join('./datasets', self.dataset_name+'.npz'), 
                    body_acc=self.body_acc, 
                    tire_acc=self.tire_acc, 
                    body_accf = self.body_accf,
                    tire_accf = self.tire_accf,
                    body_vel=self.body_vel,
                    tire_vel=self.tire_vel,
                    def_vel=self.def_vel,
                    a=self.a, 
                    G=self.G, 
                    K=self.K, 
                    u=self.u,
                    r=self.r 
                )
    def read(self, window_size):
        # 'Read' in this example code convert the dataset into windowed data.
        # Not necessary if you are not dealing with sequential data or model.
        dataset = np.load(os.path.join('./datasets', self.dataset_name+'.npz')

        self.body_acc = dataset['body_acc']
        self.tire_acc = dataset['tire_acc']
        self.body_vel = dataset['body_vel']
        self.tire_vel = dataset['tire_vel']
        self.def_vel = dataset['def_vel']
        self.body_accf = dataset['body_accf']
        self.tire_accf = dataset['tire_accf']
        self.a = dataset['a']
        self.G = dataset['G']
        self.K = dataset['K']
        self.u = dataset['u']
        self.r = dataset['r']

        windowed_body_acc = []
        windowed_tire_acc = []
        windowed_body_vel = []
        windowed_tire_vel = []
        windowed_def_vel = []
        windowed_body_accf = []
        windowed_tire_accf = []
        windowed_a = []
        windowed_a_seq = []
        windowed_K = []
        windowed_K_seq = []
        windowed_G = []
        windowed_G_seq = []
        windowed_u = []
        windowed_r = []

        for episode in range(int(self.u.shape[0] / data_per_episode)):
            for step in range(data_per_episode):
                data_idx = data_per_episode*episode+step

                body_acc_seq = []
                tire_acc_seq = []
                body_vel_seq = []
                tire_vel_seq = []
                def_vel_seq = []
                body_accf_seq = []
                tire_accf_seq = []
                a_seq = []
                G_seq = []
                K_seq = []
                u_seq = []
                r_seq = []

                if step < window_size - 1:
                    pass
                else:
                    for window_shift in range(window_size):
                        # At t_n, data from t_{n-w+1} to t_n are stored together.
                        _ba = self.body_acc[data_idx-window_shift]
                        _ta = self.tire_acc[data_idx-window_shift]
                        _bv = self.body_vel[data_idx-window_shift]
                        _tv = self.tire_vel[data_idx-window_shift]
                        _df = self.def_vel[data_idx-window_shift]
                        _baf = self.body_accf[data_idx-window_shift]
                        _taf = self.tire_accf[data_idx-window_shift]
                        _a = self.a[data_idx-window_shift]
                        _K = self.K[data_idx-window_shift]
                        _G = self.G[data_idx-window_shift]
                        _u = self.u[data_idx-window_shift]
                        _r = self.r[data_idx-window_shift]

                        body_acc_seq.append(_ba)
                        tire_acc_seq.append(_ta)
                        body_vel_seq.append(_bv)
                        tire_vel_seq.append(_tv)
                        def_vel_seq.append(_df)
                        body_accf_seq.append(_taf)
                        tire_accf_seq.append(_baf)
                        
                        a_seq.append(_a)
                        K_seq.append(_K)
                        G_seq.append(_G)
                        u_seq.append(_u)
                        r_seq.append(_r)
                    
                    windowed_body_acc.append(body_acc_seq)
                    windowed_tire_acc.append(tire_acc_seq)
                    windowed_body_vel.append(body_vel_seq)
                    windowed_tire_vel.append(tire_vel_seq)
                    windowed_def_vel.append(def_vel_seq)
                    windowed_body_accf.append(body_accf_seq)
                    windowed_tire_accf.append(tire_accf_seq)
                    windowed_a.append(self.a[data_idx])
                    windowed_a_seq.append(a_seq)
                    windowed_K.append(self.K[data_idx])
                    windowed_K_seq.append(K_seq)
                    windowed_G.append(self.G[data_idx])
                    windowed_G_seq.append(K_seq)
                    windowed_u.append(u_seq)
                    windowed_r.append(r_seq)

        np.savez('./datasets/'+self.dataset_name+'_'+str(window_size)+'.npz', 
                body_acc=windowed_body_acc, 
                tire_acc=windowed_tire_acc, 
                body_vel=windowed_body_vel,
                tire_vel=windowed_tire_vel,
                def_vel=windowed_def_vel,
                body_accf=windowed_body_accf,
                tire_accf=windowed_tire_accf,
                a=windowed_a, 
                a_seq=windowed_a_seq, 
                K=windowed_K,
                r=windowed_r,
                K_seq=windowed_K_seq, 
                G=windowed_G,
                G_seq=windowed_G_seq, 
                u=windowed_u, 
            )

        return {
                'ba':   np.asarray(windowed_body_acc),
                'ta':   np.asarray(windowed_tire_acc),
                'bv':   np.asarray(windowed_body_vel),
                'tv':   np.asarray(windowed_tire_vel),
                'dv':   np.asarray(windowed_def_vel),
                'baf':  np.asarray(windowed_body_accf),
                'taf':  np.asarray(windowed_tire_accf),
                'a':    np.asarray(windowed_a),
                'a_seq':np.asarray(windowed_a_seq),
                'K':    np.asarray(windowed_K),                
                'K_seq':np.asarray(windowed_K_seq),
                'G':    np.asarray(windowed_G),
                'G_seq':np.asarray(windowed_G_seq),
                'u':    np.asarray(windowed_u),
                'r':    np.asarray(windowed_r)
            }

    def load(self, window_size):
        # Loads saved (windowed) numpy dataset.
        # the variable window_size is not necessary if you are not dealing with sequential data or model.
        d = np.load('./datasets/'+self.dataset_name+'_'+str(window_size)+'.npz')

        self.body_acc   = d['body_acc']
        self.tire_acc   = d['tire_acc']
        self.body_vel   = d['body_vel']
        self.tire_vel   = d['tire_vel']
        self.def_vel    = d['def_vel']
        self.body_accf  = d['body_accf']
        self.tire_accf  = d['tire_accf']
        self.a = d['a']
        self.G = d['G']
        self.K = d['K']
        self.u = d['u']
        self.r = d['r']

        return {
            'ba':   d['body_acc'],
            'ta':   d['tire_acc'],
            'bv':   d['body_vel'],
            'tv':   d['tire_vel'],
            'dv':   d['def_vel'],
            'taf':  d['body_accf'],
            'baf':  d['tire_accf'],
            'a':    d['a'],
            'a_seq':d['a_seq'],
            'G':    d['G'],
            'G_seq':d['G_seq'],
            'K':    d['K'],
            'K_seq':d['K_seq'],
            'u':    d['u'],
            'r':    d['r']
        }

if __name__ == "__main__":
    dl = DataLoader()
    dl.create()
    dl.read(80)
    dl.load(80)
