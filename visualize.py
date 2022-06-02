# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# dir="/home/fengkai/ROB590/minicheetah-traversability-irl/SVF_map/"
# # dir="/home/fengkai/ROB590/minicheetah-traversability-irl/height_map/"
# save_dir="/home/fengkai/ROB590/minicheetah-traversability-irl/SVF_map/figure/"
# for i in range(240):
#     reward=np.loadtxt(dir+'SVF_map_{step:02d}.txt'.format(step=i))
#     plt.figure()
#     sns.heatmap(reward, cmap='viridis')
#     plt.savefig(save_dir+'Figure_{step:02d}'.format(step=i),format="png")
# for i in range(240):
#     h=np.loadtxt(dir+'height_map_{step:02d}.txt'.format(step=i))
# sns.heatmap(h, cmap='viridis')
# plt.show()
# plt.savefig(save_dir+'Figure_{step:02d}'.format(step=i),format="png")

import numpy as np
import warnings
import logging
import os
import torch
import time
from network.reward_net1 import RewardNet
from network.res_unet import ResUnet
from torch.utils.data import DataLoader
import loader.sim_data_loader as data_loader
import visdom
import mdp.offroad_grid as offroad_grid
import loader.offroad_loader as offroad_loader

from maxent_nonlinear_offroad import pred, rl, overlay_traj_to_map, visualize
from network.hybrid_dilated import HybridDilated

net=ResUnet(channel=3)
input=torch.zeros((1,3,80,80))
out=net(input)
print(out.shape)