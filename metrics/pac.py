'''
    Metrics: 
        PAC Bayesian flatness with respect to input and weight

'''

import os 
import time
import copy 
import torch 
import numpy as np
import torch.nn.functional as F

from dlrm_s_pytorch import unpack_batch

__all__ = ['eval_pac_weight', 'eval_pac_input']

@torch.no_grad()
def evaluate_function_noise(xloader, dlrm, network, noise, loss_fn_wrap, use_gpu=True, ndevices=1):
    #eval function for weight perturbation
    dlrm.eval()
    num_batch = 50
    device = torch.cuda.current_device()
    sum_E = 0
    for i, inputBatch in enumerate(xloader):
        if num_batch > 0 and i >= num_batch: break
        X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)
        gaussian_noise = noise * torch.randn_like(X)
        #new_X = (X + gaussian_noise).clamp(0, 1)

        # compute output
        Z = network(
            new_X,
            lS_o,
            lS_i,
            use_gpu,
            device,
            ndevices=ndevices,
        )
        E = loss_fn_wrap(Z, T, use_gpu, device)
        sum_E += E.detach().cpu().numpy()

    return sum_E / num_batch
@torch.no_grad()
def evaluate_function(xloader, dlrm, network, loss_fn_wrap, use_gpu=True, ndevices=1):
    #eval function for weight perturbation
    dlrm.eval()
    num_batch = 50
    device = torch.cuda.current_device()
    sum_E = 0
    for i, inputBatch in enumerate(xloader):
        if num_batch > 0 and i >= num_batch: break
        X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)

        # compute output
        Z = network(
            X,
            lS_o,
            lS_i,
            use_gpu,
            device,
            ndevices=ndevices,
        )
        E = loss_fn_wrap(Z, T, use_gpu, device)
        sum_E += E.detach().cpu().numpy()

    return sum_E / num_batch


def eval_pac_weight(
    dlrm, xloader, network, loss_fn_wrap, train_mode=False, num_batch=5, use_gpu=True, ndevices=1,
    beta=0.1, 
    max_search_times=20,
    iteration_times=15, 
    sigma_min=0., 
    sigma_max=5., 
    eps=1e-3):

    original_weight = copy.deepcopy(dlrm.state_dict())
    device = torch.cuda.current_device()

    original_loss = evaluate_function(xloader, dlrm, network, loss_fn_wrap, use_gpu=use_gpu, ndevices=ndevices) # numpy array
    max_loss = (1 + beta) * original_loss

    for episode in range(max_search_times):
    
        sigma_new = (sigma_max + sigma_min) / 2
    
        loss_list = []
        for step in range(iteration_times):
        # generate perturbed weight 
            perturb_weight = {}
            for key in original_weight.keys():
                if 'mask' in key:
                # mask that represents network structure.
                    perturb_weight[key] = original_weight[key]
                else:
                    if len(original_weight[key].size()) in [2,4]:
                        perturb_weight[key] = torch.normal(mean = original_weight[key], std = sigma_new * (original_weight[key].abs()))
                    else:
                        perturb_weight[key] = original_weight[key]
                        
            dlrm.load_state_dict(perturb_weight)
            perturb_loss = evaluate_function(xloader, dlrm, network, loss_fn_wrap, use_gpu=use_gpu, ndevices=ndevices)
            loss_list.append(perturb_loss)  

        loss_mean = np.mean(np.array(loss_list))
        print('current-sigma = {}, tolerent loss = {}, current loss = {}'.format(sigma_new, max_loss, loss_mean))
        #compare with original_loss 
        if loss_mean <= max_loss and (sigma_max - sigma_min) < eps:
            return sigma_new
        else:
            if loss_mean > max_loss:
                sigma_max = sigma_new
            else:
                sigma_min = sigma_new

    dlrm.load_state_dict(original_weight)
    return 1 / sigma_new**2

def eval_pac_input(
    dlrm, xloader, network, loss_fn_wrap, train_mode=False, num_batch=5, use_gpu=True, ndevices=1,
    beta=0.1, 
    max_search_times=20,
    iteration_times=15, 
    sigma_min=0., 
    sigma_max=5., 
    eps=1e-3):

    original_weight = copy.deepcopy(dlrm.state_dict())
    device = torch.cuda.current_device()
    original_loss =  evaluate_function(xloader, dlrm, network, loss_fn_wrap, use_gpu=use_gpu, ndevices=ndevices) # numpy array
    max_loss = (1 + beta) * original_loss

    for episode in range(max_search_times):
    
        sigma_new = (sigma_max + sigma_min) / 2
    
        loss_list = []
        for step in range(iteration_times):
            perturb_loss = evaluate_function_noise(xloader, dlrm, network, sigma_new, loss_fn_wrap, use_gpu=True, ndevices=1)
            loss_list.append(perturb_loss)  

        loss_mean = np.mean(np.array(loss_list))
        print('current-sigma = {}, tolerent loss = {}, current loss = {}'.format(sigma_new, max_loss, loss_mean))
        #compare with original_loss 
        if loss_mean <= max_loss and (sigma_max - sigma_min) < eps:
            return sigma_new
        else:
            if loss_mean > max_loss:
                sigma_max = sigma_new
            else:
                sigma_min = sigma_new
    dlrm.load_state_dict(original_weight)
    return 1 / sigma_new**2


