import logging
from matplotlib.pyplot import grid
from torch.utils.data import Dataset
import os
import scipy.io as sio

import numpy as np
from scipy import optimize

np.set_printoptions(threshold=np.inf, suppress=True)
import random
import copy
import math


class SimLoader(Dataset):
    def __init__(self, grid_size, train=True, demo=None, datadir='/home/fengkai/ROB590/IIG-IRL/data/', pre_train=False, tangent=False,
                 more_kinematic=None):
        assert grid_size % 2 == 0, "grid size must be even number"
        self.grid_size = grid_size
        if train:
            self.data_dir = datadir + 'data_v/'
            # self.data_dir = datadir + 'train_data/'
        else:
            self.data_dir = datadir + 'data_vt/'
        items = os.listdir(self.data_dir)
        self.data_list = []
        for item in items:
            self.data_list.append(self.data_dir + '/' + item)

        # self.pre_train = pre_train

        # kinematic related feature
        self.center_idx = self.grid_size / 2
        self.delta_x_layer = np.zeros((self.grid_size, self.grid_size), dtype=np.float)
        self.delta_y_layer = self.delta_x_layer.copy()

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.delta_x_layer[x, y] = x - self.center_idx
                self.delta_y_layer[x, y] = y - self.center_idx

    def __getitem__(self, index):
        
        feat,past_traj,future_traj,average_volume=np.genfromtxt(self.data_list[index]+"/map.csv"),np.genfromtxt(self.data_list[index]+"/past_traj.csv"),np.genfromtxt(self.data_list[index]+"/future_traj.csv"),np.genfromtxt(self.data_list[index]+"/average_volume.csv")
        # feat,past_traj,future_traj=np.genfromtxt(self.data_list[index]+"/map.csv"),np.genfromtxt(self.data_list[index]+"/past_traj.csv"),np.genfromtxt(self.data_list[index]+"/future_traj.csv")
        feat=feat.reshape((1,self.grid_size,self.grid_size))
        normalization = 0.5 * self.grid_size
        feat = np.vstack((feat, np.expand_dims(self.delta_x_layer.copy() / normalization, axis=0)))
        feat = np.vstack((feat, np.expand_dims(self.delta_y_layer.copy() / normalization, axis=0)))

        feat[0] = (feat[0] - np.mean(feat[0])) / np.std(feat[0])

        future_traj = self.auto_pad_future(future_traj[:, :2])
        past_traj = self.auto_pad_past(past_traj[:, :2])
    

        return feat, past_traj, future_traj,average_volume
    def __len__(self):
        return len(self.data_list)
        
    def auto_pad_past(self, traj):
            """
            add padding (NAN) to traj to keep traj length fixed.
            traj shape needs to be fixed in order to use batch sampling
            :param traj: numpy array. (traj_len, 2)
            :return:
            """
            fixed_len = self.grid_size
            if traj.shape[0] >= self.grid_size:
                traj = traj[traj.shape[0]-self.grid_size:, :]
                #raise ValueError('traj length {} must be less than grid_size {}'.format(traj.shape[0], self.grid_size))
            pad_len = self.grid_size - traj.shape[0]
            pad_array = np.full((pad_len, 2), np.nan)
            output = np.vstack((traj, pad_array))
            return output

    def auto_pad_future(self, traj):
            """
            add padding (NAN) to traj to keep traj length fixed.
            traj shape needs to be fixed in order to use batch sampling
            :param traj: numpy array. (traj_len, 2)
            :return:
            """
            fixed_len = self.grid_size
            if traj.shape[0] >= self.grid_size:
                traj = traj[:self.grid_size, :]
                #raise ValueError('traj length {} must be less than grid_size {}'.format(traj.shape[0], self.grid_size))
            pad_len = self.grid_size - traj.shape[0]
            pad_array = np.full((pad_len, 2), np.nan)
            output = np.vstack((traj, pad_array))
            return output