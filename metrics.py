import numpy as np
import torch

from dlrm_s_pytorch import unpack_batch

def get_ntk_n(dlrm, xloader, network, train_mode=False, num_batch=5, use_gpu=True, ndevices=1):
    device = torch.cuda.current_device()
    #network.eval()
    ######
    grads = []
    for i, inputBatch in enumerate(xloader):
        if num_batch > 0 and i >= num_batch: break
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
        if isinstance(logit, tuple):
            logit = logit[1]  # 201 networks: return features and logits
        for _idx in range(len(X)):
            logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
            grad = []
            for name, W in dlrm.named_parameters():
                if W.grad is not None:
                    try:
                        grad.append(W.grad.view(-1).detach())
                    except:
                        #grad.append(W.grad.coalesce().values().view(-1).detach())
                        pass

            grads.append(torch.cat(grad, -1))
            dlrm.zero_grad()
            torch.cuda.empty_cache()
    ######
    grads = torch.stack(grads, 0)
    ntk = torch.einsum('nc,mc->nm', [grads, grads])
    eigenvalues, _ = torch.symeig(ntk)  # ascending
    cond = (np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0))
    return cond