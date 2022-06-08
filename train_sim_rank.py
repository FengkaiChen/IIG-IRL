import numpy as np
import warnings
import logging
import os
import torch
import time

from tqdm import trange
from network.reward_net import RewardNet
from torch.utils.data import DataLoader
import loader.sim_data_loader as data_loader
import visdom
import mdp.offroad_grid as offroad_grid
from network.res_unet import ResUnet
from torch.autograd import Variable

from maxent_nonlinear_offroad_sim import pred, pred_rank, rl, overlay_traj_to_map, visualize

from network.hybrid_dilated import HybridDilated

pre_train_weight = None
vis_per_steps = 5
test_per_steps = 5
resume = None

grid_size = 80
discount = 0.9
lr = 1e-4
n_epoch = 32
batch_size = 8
n_worker = 8
use_gpu = True
best_test_nll = np.inf
best_test_loss = np.inf
model = offroad_grid.OffroadGrid(grid_size, discount)
n_states = model.n_states
n_actions = model.n_actions

vis = visdom.Visdom(env='main')


exp_name = 'Matterport_rank'
# net = HybridDilated(feat_out_size=25, regression_hidden_size=64)
# net = RewardNet(n_channels=3,n_classes=1)
net=ResUnet(channel=3)
# exp_name = 'ResUnet'
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
loss_criterion = torch.nn.CrossEntropyLoss()


for epoch in range(n_epoch):
    for i,(feat, past_traj, future_traj,average_volume) in enumerate(train_loader):
        # feat=torch.tensor(feat)
        net.train()
        # print(feat.shape,past_traj.shape, future_traj.shape)

        start = time.time()
        # print(feat.shape, past_traj.shape, future_traj.shape,average_volume.shape)
        nll_list, r_var, svf_diff_var, values_list, past_return_var = pred_rank(feat, future_traj,past_traj,net, n_states, model, grid_size)
        opt.zero_grad()
        torch.autograd.backward([r_var], [-svf_diff_var])  # to maximize, hence add minus sign
        
        # Trajectory Ranking
        half_batch_size = past_return_var.shape[0] // 2
        past_return_var_i = past_return_var[:half_batch_size]
        past_return_var_j = past_return_var[half_batch_size:half_batch_size*2]
        output = torch.cat((past_return_var_i.unsqueeze(dim=1), past_return_var_j.unsqueeze(dim=1)), dim=1)
        ave_energy_cons_i = average_volume[:half_batch_size]
        ave_energy_cons_j = average_volume[half_batch_size:half_batch_size*2]
        target = torch.gt(ave_energy_cons_i, ave_energy_cons_j).squeeze().long() # 0 when i is better, 1 when j is better
        
        loss = loss_criterion(output, target)
        loss_var = Variable(loss, requires_grad=True)
        loss_var.backward()
        

        opt.step()
        nll = sum(nll_list) / len(nll_list)
        print('main. acc {}. loss {}. took {} s'.format(nll, loss, time.time() - start))

        if step % vis_per_steps == 0 or nll > 2.5:
            # visualize(past_traj, future_traj, feat, r_var, values_list, svf_diff_var, step, vis, grid_size, train=True)
            if step == 0:
                step += 1
                continue
        if step % test_per_steps == 0:
            # test
            net.eval()
            nll_test_list = []
            loss_test_list = []
            for _, (feat, past_traj, future_traj,average_volume) in enumerate(test_loader):
                tmp_nll, r_var, svf_diff_var, values_list, past_return_var = pred_rank(feat, future_traj, past_traj, net, n_states, model, grid_size)
                nll_test_list += tmp_nll

                half_batch_size = past_return_var.shape[0] // 2
                past_return_var_i = past_return_var[:half_batch_size]
                past_return_var_j = past_return_var[half_batch_size:half_batch_size*2]
                output = torch.cat((past_return_var_i.unsqueeze(dim=1), past_return_var_j.unsqueeze(dim=1)), dim=1)
                ave_energy_cons_i = average_volume[:half_batch_size]
                ave_energy_cons_j = average_volume[half_batch_size:half_batch_size*2]
                target = torch.gt(ave_energy_cons_i, ave_energy_cons_j).squeeze().long() # 0 when i is better, 1 when j is better
                tmp_loss = loss_criterion(output, target)
                loss_test_list.append(tmp_loss)

            nll_test = sum(nll_test_list) / len(nll_test_list)
            loss_test = sum(loss_test_list) / len(loss_test_list)
            print('main. test nll {}. test loss {}'.format(nll_test, loss_test))
            visualize(past_traj, future_traj, feat, r_var, values_list, svf_diff_var, step, vis, grid_size, train=False)
            # if nll_test < best_test_nll:
            #         best_test_nll = nll_test
            #         state = {'nll_cma': nll_cma, 'test_nll': nll_test, 'loss_cma': loss_cma, 'test_loss': loss_test, 'step': step, 'net_state': net.state_dict(),
            #                 'opt_state': opt.state_dict(), 'discount':discount}
            #         path = os.path.join('exp', exp_name, 'step{}-nll{}-loss{}-total{}.pth'.format(step, nll_test, loss_test, nll_test+loss_test))
            #         torch.save(state, path)

            # if loss_test < best_test_loss:
            #         best_test_loss = loss_test
            #         state = {'nll_cma': nll_cma, 'test_nll': nll_test, 'loss_cma': loss_cma, 'test_loss': loss_test, 'step': step, 'net_state': net.state_dict(),
            #                 'opt_state': opt.state_dict(), 'discount':discount}
            #         path = os.path.join('exp', exp_name, 'step{}-nll{}-loss{}-total{}.pth'.format(step, nll_test, loss_test, nll_test+loss_test))
        
        step += 1

