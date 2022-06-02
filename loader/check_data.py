import os
import numpy as np
import scipy.io as sio

datadir='/home/fengkai/ROB590/minicheetah_irldata/training_data/'
items = os.listdir(datadir)
data_list = []
for item in items:
    data_list.append(datadir + '/' + item)
data_mat = sio.loadmat(data_list[0])
feat, robot_state_feat, past_traj, future_traj, ave_energy_cons = data_mat['feat'].copy(), data_mat['robot_state_data'], data_mat['past_traj'], data_mat['future_traj'], data_mat['average_energy_consumption']
print(past_traj.shape,future_traj.shape)