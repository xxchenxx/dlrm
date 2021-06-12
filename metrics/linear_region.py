import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset
from pdb import set_trace as bp
from operator import mul
from functools import reduce
from dlrm_s_pytorch import unpack_batch


import copy
def linear_region(dlrm, xloader, network, train_mode=False, num_batch=5, use_gpu=True, ndevices=1):
    #dlrm_clone = copy.deepcopy(dlrm)
    device = torch.cuda.current_device()
    features = []
    def hook_in_forward(module, input, output):
        features.append(output.detach())
    handles = []
    for m in dlrm.modules():
        if isinstance(m, nn.ReLU): #or isinstance(m, nn.Sigmoid):
            handles.append(m.register_forward_hook(hook=hook_in_forward))
    
    loader = iter(xloader)
    inputBatch = loader.next()
    X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)
    dlrm.zero_grad()
    logit = network(
        X,
        lS_o,
        lS_i,
        use_gpu,
        device,
        ndevices=ndevices,
    )
    features = torch.cat([f.view(f.size(0), -1).cuda('cuda:0') for f in features], 1)
    features = features.view(features.shape[0] * 2, -1)
    activations = torch.sign(features)  # after ReLU
    res = torch.matmul(activations.half(), (1 - activations).T.half())
    res += res.T
    res = 1 - torch.sign(res)
    res = res.sum(1)
    res = 1. / res.float()
    res = res.sum().item()
    for handle in handles:
        handle.remove()
    return res