import numpy as np
import warnings
import logging
import os
import torch
import time
from network.reward_net import RewardNet
from torch.utils.data import DataLoader
import loader.sim_data_loader as data_loader
import visdom
import mdp.offroad_grid as offroad_grid
from network.res_unet import ResUnet


from maxent_nonlinear_offroad import pred, rl, overlay_traj_to_map, visualize
from network.hybrid_dilated import HybridDilated

pre_train_weight = None
vis_per_steps = 10
test_per_steps = 10
resume = None

grid_size = 80
discount = 0.9
lr = 1e-4
n_epoch = 32
batch_size = 8
n_worker = 8
use_gpu = True
best_test_nll = np.inf

model = offroad_grid.OffroadGrid(grid_size, discount)
n_states = model.n_states
n_actions = model.n_actions

vis = visdom.Visdom(env='main')

# net = RewardNet(n_channels=3,n_classes=1)
# exp_name = 'Matterport'
# net = HybridDilated(feat_out_size=25, regression_hidden_size=64)
net=ResUnet(channel=3)
exp_name = 'ResUnet'
if not os.path.exists(os.path.join('exp', exp_name)):
    os.makedirs(os.path.join('exp', exp_name))


train_loader = data_loader.SimLoader(grid_size=grid_size, tangent=False, more_kinematic=None)
train_loader = DataLoader(train_loader, num_workers=n_worker, batch_size=batch_size, shuffle=True)

test_loader = data_loader.SimLoader(grid_size=grid_size, train=False, tangent=False)
test_loader = DataLoader(test_loader,num_workers=n_worker, batch_size=batch_size, shuffle=True)



step = 0
nll_cma = 0
nll_test = 0
loss_cma = 0
loss_test = 0
opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

# opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
# loss_criterion = torch.nn.CrossEntropyLoss()

for epoch in range(n_epoch):
    for i,(feat, past_traj, future_traj) in enumerate(train_loader):
        # feat=torch.tensor(feat)
        net.train()
        print(feat.shape,past_traj.shape, future_traj.shape)
        start = time.time()
        nll_list, r_var, svf_diff_var, values_list = pred(feat, future_traj, net, n_states, model, grid_size)
        opt.zero_grad()
        torch.autograd.backward([r_var], [-svf_diff_var])
        opt.step()
        nll = sum(nll_list) / len(nll_list)
        print('main. acc {}. took {} s'.format(nll, time.time() - start))
        

        if step % vis_per_steps == 0 or nll > 2.5:
            visualize(past_traj, future_traj, feat, r_var, values_list, svf_diff_var, step, vis, grid_size, train=True)
            if step == 0:
                step += 1
                continue
        if step % test_per_steps == 0:
            # test
            net.eval()
            nll_test_list = []
            for _, (feat, past_traj, future_traj) in enumerate(test_loader):
                tmp_nll, r_var, svf_diff_var, values_list = pred(feat, future_traj, net, n_states, model, grid_size)
                nll_test_list += tmp_nll
            nll_test = sum(nll_test_list) / len(nll_test_list)
            print('main. test nll {}'.format(nll_test))
            if nll_test < best_test_nll:
                best_test_nll = nll_test
                state = {'nll_cma': nll_cma, 'test_nll': nll_test, 'step': step, 'net_state': net.state_dict(),
                         'opt_state': opt.state_dict(), 'discount':discount}
                path = os.path.join('exp', exp_name, 'step{}-loss{}.pth'.format(step, nll_test))
                torch.save(state, path)
            

        
        step += 1

