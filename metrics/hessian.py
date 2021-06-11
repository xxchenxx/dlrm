#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

import torch
import math
from torch.autograd import Variable
import numpy as np


def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def group_add(params, update, alpha=1):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :return:
    """
    for i, p in enumerate(params):
        params[i].data.add_(update[i] * alpha)
    return params


def normalization(v):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    s = group_product(v, v)
    s = s**0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    return v


def get_params_grad(model):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    for name, param in model.named_parameters():
        if not param.requires_grad or 'emb' in name:
            continue
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
        
    return params, grads


def hessian_vector_product(gradsH, params, v):
    """
    compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    """
    hv = torch.autograd.grad(gradsH,
                             params,
                             grad_outputs=v,
                             only_inputs=True,
                             retain_graph=True)
    return hv


def orthnormal(w, v_list):
    """
    make vector w orthogonal to each vector in v_list.
    afterwards, normalize the output w
    """
    for v in v_list:
        w = group_add(w, v, alpha=-group_product(w, v))
    return normalization(w)

from dlrm_s_pytorch import unpack_batch

def hessian_eigen(
    dlrm, xloader, network, loss_fn_wrap, train_mode=False, num_batch=5, use_gpu=True, ndevices=1,
    beta=0.1, 
    max_search_times=20,
    iteration_times=15, 
    sigma_min=0., 
    sigma_max=5., 
    eps=1e-3):
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
        E.backward(create_graph=True)

        eigenvalues = []
        eigenvectors = []

        computed_dim = 0

        params, gradsH = get_params_grad(dlrm)
        v = []
        for name, p in dlrm.named_parameters():
            if not 'emb' in name:
                v.append(torch.randn(p.size()).to(device))
                
        v = normalization(v)  # normalize the vector
        eigenvalue = None
        for i in range(100):
            v = orthnormal(v, eigenvectors)
            dlrm.zero_grad()
            
            Hv = hessian_vector_product(gradsH, params, v)
            tmp_eigenvalue = group_product(Hv, v).cpu().item()

            v = normalization(Hv)

            if eigenvalue == None:
                eigenvalue = tmp_eigenvalue
            else:
                if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) +
                                                       1e-6) < 1e-3:
                    break
                else:
                    eigenvalue = tmp_eigenvalue
        eigenvalues.append(eigenvalue)
        eigenvectors.append(v)
        computed_dim += 1
    print(eigenvalues)
    return eigenvalues, eigenvectors

def hessian_trace(
    dlrm, xloader, network, loss_fn_wrap, train_mode=False, num_batch=5, use_gpu=True, ndevices=1,
    beta=0.1, 
    max_search_times=20,
    iteration_times=15, 
    sigma_min=0., 
    sigma_max=5., 
    eps=1e-3):
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
        E.backward(create_graph=True)

        eigenvalues = []
        eigenvectors = []

        computed_dim = 0

        params, gradsH = get_params_grad(dlrm)
        trace_vhv = []
        trace = 0
        

        for i in range(100):
            dlrm.zero_grad()
            #params, gradsH = get_params_grad(dlrm)
            v = []
            for name, p in dlrm.named_parameters():
                if not 'emb' in name:
                    v.append(torch.randn(p.size()).to(device))
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1
            
            Hv = hessian_vector_product(gradsH, params, v)
            trace_vhv.append(group_product(Hv, v).cpu().item())

            if abs(np.mean(trace_vhv) - trace) / (trace + 1e-6) < 1e-3:
                return trace_vhv
            else:
                trace = np.mean(trace_vhv)

    return trace


def hessian_eigen_input(
    dlrm, xloader, network, loss_fn_wrap, train_mode=False, num_batch=5, use_gpu=True, ndevices=1,
    beta=0.1, 
    max_search_times=20,
    iteration_times=15, 
    sigma_min=0., 
    sigma_max=5., 
    eps=1e-3):
    device = torch.cuda.current_device()
    #network.eval()
    ######
    grads = []
    for i, inputBatch in enumerate(xloader):
        if num_batch > 0 and i >= num_batch: break
        X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)
        X.requires_grad=True
        Z = network(
            X,
            lS_o,
            lS_i,
            use_gpu,
            device,
            ndevices=ndevices,
        )
        E = loss_fn_wrap(Z, T, use_gpu, device)
        E.backward(create_graph=True)

        eigenvalues = []
        eigenvectors = []

        computed_dim = 0

        params, gradsH = [X], [X.grad + 0.]
        
        v = [torch.randn(p.size()).to(device) for p in dlrm.parameters()
            ]  # generate random vector
        v = normalization(v)  # normalize the vector

        for i in range(10):
            v = orthnormal(v, eigenvectors)
            dlrm.zero_grad()
            #X.zero_grad()
            Hv = hessian_vector_product(gradsH, params, v)
            tmp_eigenvalue = group_product(Hv, v).cpu().item()

            v = normalization(Hv)

            if eigenvalue == None:
                eigenvalue = tmp_eigenvalue
            else:
                if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) +
                                                       1e-6) < 1e-3:
                    break
                else:
                    eigenvalue = tmp_eigenvalue
        eigenvalues.append(eigenvalue)
        eigenvectors.append(v)
        computed_dim += 1

    return eigenvalues, eigenvectors


def hessian_trace_input(
    dlrm, xloader, network, loss_fn_wrap, train_mode=False, num_batch=5, use_gpu=True, ndevices=1,
    beta=0.1, 
    max_search_times=20,
    iteration_times=15, 
    sigma_min=0., 
    sigma_max=5., 
    eps=1e-3):
    device = torch.cuda.current_device()
    #network.eval()
    ######
    grads = []
    for i, inputBatch in enumerate(xloader):
        if num_batch > 0 and i >= num_batch: break
        X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)
        X.requires_grad = True
        Z = network(
            X,
            lS_o,
            lS_i,
            use_gpu,
            device,
            ndevices=ndevices,
        )
        E = loss_fn_wrap(Z, T, use_gpu, device)
        E.backward(create_graph=True)

        computed_dim = 0

        #params, gradsH = get_params_grad(dlrm)
        params = [X]
        gradsH = [X.grad + 0.]
        trace_vhv = []
        trace = 0
        

        for i in range(100):
            dlrm.zero_grad()
            #X.zero_grad()
            v = [
                torch.randint_like(p, high=2, device=device)
                for p in params
            ]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1
            
            Hv = hessian_vector_product(gradsH, params, v)
            trace_vhv.append(group_product(Hv, v).cpu().item())

            if abs(np.mean(trace_vhv) - trace) / (trace + 1e-6) < 1e-3:
                return trace_vhv
            else:
                trace = np.mean(trace_vhv)

    return trace_vhv