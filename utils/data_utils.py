# Laboratory of Robotics and Cognitive Science
# Version by:  Rafael Anicet Zanini
# Github:      https://github.com/larocs/EMG-GAN

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical


class DataLoader():
    def __init__(self, args):
        self.file_path = args['training_file']
        self.features = args['features']
        self.labels = np.array(args["labels"])
        self.channels = len(self.features)
        self.rescale = args['rescale']
        self.num_steps = args['num_steps']
        self.train_split = args['train_split']
        self.batch_size = args['batch_size']
    
    def load_training_data(self, to_cat: bool=True):
        data = self.load_timeseries(self.file_path, self.features)

        #Group values by timestamp to get ACC-burts
        grouped = data.groupby(["timestamp"])

        # Initialize empty array
        X = np.zeros((0, self.num_steps, self.channels))
        Y = []
        for _, ep in grouped:
            acc_values = ep[self.features].values[None,:self.num_steps,...]
            # Bursts need to be at least the same size as "self.num_steps", so we pad with zeroes
            if acc_values.shape[1] < self.num_steps:
                pad_size = self.num_steps - acc_values.shape[1]
                npad = ((0, 0), (0, pad_size), (0, 0))
                acc_values = np.pad(acc_values, pad_width=npad, mode='constant') 

            X = np.vstack((X, acc_values)) # Feature data
           # Y.append(np.argwhere(self.labels==ep.Bev.values[0])[0][0]) # Labels

        # # Convert Y to categorical
        # Y = np.array(Y)
        # if to_cat: Y = to_categorical(Y, num_classes=len(self.labels))
        
        return X #, Y
    
    def load_timeseries(self, filename, series):
        #Load time series dataset
        loaded_series = pd.read_csv(filename, sep=',', header=0, index_col=0, squeeze=True)
       
        #Applying filter on the selected series
        selected_series = loaded_series.filter(items=series + ["timestamp", "Bev"])

        return selected_series
    
    def min_max(self, data, min, max):
        """Normalize data"""
        scaler = MinMaxScaler(feature_range=(min, max),copy=True)
        scaler.fit(data)
        norm_value = scaler.transform(data)
        return [norm_value, scaler]
    
    # def get_windows(self, data, window_size):
    #     # Split data into windows
    #     raw = []
    #     for index in range(len(data) - window_size):
    #         raw.append(data[index: index + window_size])
    #     return raw
    
    def normalize(self, data):
        """Normalize data"""
        scaler = MinMaxScaler(feature_range=(0, 1),copy=True)
        scaler.fit(data)
        norm_value = scaler.transform(data)
        return [norm_value, scaler]
    
    def get_training_batch(self):
        x_train = self.load_training_data()
        idx = np.random.randint(0, x_train.shape[0], self.batch_size)
        signals = x_train[idx]
        signals = np.reshape(signals, (signals.shape[0], signals.shape[1], self.channels))
        return signals