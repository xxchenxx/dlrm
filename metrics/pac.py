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
    num_batch = 10
    device = torch.cuda.current_device()
    sum_E = 0
    for i, inputBatch in enumerate(xloader):
        if num_batch > 0 and i >= num_batch: break
        X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)
        gaussian_noise = noise * torch.randn_like(X)
        new_image = (X + gaussian_noise).clamp(0, 1)

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
        break

    return sum_E / num_batch



def eval_pac_weight(
    dlrm, xloader, network, loss_fn_wrap, train_mode=False, num_batch=5, use_gpu=True, ndevices=1,
    beta=0.1, 
    max_search_times=20,
    iteration_times=15, 
    sigma_min=0., 
    sigma_max=5., 
    eps=1e-3):
    '''
    Evaluation PAC Bayesian with respect to weight (Using Binary Search)
        input:
            model # model
            dataloader # data loader
            eval_function # evaluate function, with input(model, dataloader), output(train_loss)
            device = None 
            beta = 0.1 # tolerent ratio
            max_search_times = 20 # Maximum Searching times
            iteration_times = 15 # times for perturb weight for each sigma
            sigma_min = 0 # min-sigma
            sigma_max = 5 # max-sigma
            eps = 1e-3 
        output:
            mu = 1 / (sigma ** 2)
    '''

    original_weight = copy.deepcopy(dlrm.state_dict())
    device = torch.cuda.current_device()
    #network.eval()
    ######
    grads = []
    for i, inputBatch in enumerate(xloader):
        if num_batch > 0 and i >= num_batch: break
        X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)
        Z = network(
            X,
            lS_o,
            lS_i,
            use_gpu,
            device,
            ndevices=ndevices,
        )
        E = loss_fn_wrap(Z, T, use_gpu, device)

        # compute loss and accuracy
        original_loss = E.detach().cpu().numpy()  # numpy array
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
                Z = network(
                    X,
                    lS_o,
                    lS_i,
                    use_gpu,
                    device,
                    ndevices=ndevices,
                )
                E = loss_fn_wrap(Z, T, use_gpu, device)
                perturb_loss = E.detach().cpu().numpy()
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
    '''
    Evaluation PAC Bayesian with respect to input (Using Binary Search)
        input:
            model # model
            dataloader # data loader
            eval_function_perturb # evaluate function, with input(model, dataloader), output(train_loss) add noise in input samples
            device = None
            beta = 0.1 # tolerent ratio
            max_search_times = 20 # Maximum Searching times
            iteration_times = 15 # times for perturb weight for each sigma
            sigma_min = 0 # min-sigma
            sigma_max = 5 # max-sigma
            eps = 1e-3 
        output:
            mu = 1 / (sigma ** 2)
    '''

    original_weight = copy.deepcopy(dlrm.state_dict())
    device = torch.cuda.current_device()
    #network.eval()
    ######
    grads = []
    for i, inputBatch in enumerate(xloader):
        if num_batch > 0 and i >= num_batch: break
        X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)
        Z = network(
            X,
            lS_o,
            lS_i,
            use_gpu,
            device,
            ndevices=ndevices,
        )
        E = loss_fn_wrap(Z, T, use_gpu, device)

        # compute loss and accuracy
        original_loss = E.detach().cpu().numpy()  # numpy array
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

    return 1 / sigma_new**2


