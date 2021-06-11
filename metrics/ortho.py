
import torch

def dominant_eigenvalue(A):

    N, _ = A.size()
    x = torch.rand(N, 1, device='cuda')

    # Ax = (A @ x).squeeze()
    # AAx = (A @ Ax).squeeze()

    # return torch.norm(AAx, p=2) / torch.norm(Ax, p=2)

    Ax = (A @ x)
    AAx = (A @ Ax)

    return AAx.permute(1, 0) @ Ax / (Ax.permute(1, 0) @ Ax)

def get_singular_values(A):
    ATA = A.permute(1, 0) @ A
    N, _ = ATA.size()
    largest = dominant_eigenvalue(ATA)
    I = torch.eye(N, device='cuda')  # noqa
    I = I * largest  # noqa
    tmp = dominant_eigenvalue(ATA - I)
    return tmp + largest, largest

def cond_weight(dlrm, xloader, network, train_mode=False, num_batch=5, use_gpu=True, ndevices=1):
    cond_weights = {}
    for name, weight in dlrm.named_parameters():
        if not 'emb' in name and 'weight' in name:
            s, l = get_singular_values(weight.data)
            cond_weights[name] = l / s
    
    return cond_weights

def features_dominant_eigenvalue(A):
    device = torch.cuda.current_device()
    B, N, _ = A.size()
    x = torch.randn(B, N, 1).to(device)

    for _ in range(1):
        x = torch.bmm(A, x)
    # x: 'B x N x 1'
    numerator = torch.bmm(
        torch.bmm(A, x).view(B, 1, N),
        x
    ).squeeze()
    denominator = (torch.norm(x.view(B, N), p=2, dim=1) ** 2).squeeze()

    return numerator / denominator

def features_get_singular_values(A):
    device = torch.cuda.current_device()
    AAT = torch.bmm(A, A.permute(0, 2, 1))
    B, N, _ = AAT.size()
    largest = dominant_eigenvalue(AAT)
    I = torch.eye(N).expand(B, N, N).to(device)  # noqa
    I = I * largest.view(B, 1, 1).repeat(1, N, N)  # noqa
    tmp = dominant_eigenvalue(AAT - I)
    return tmp + largest, largest

from dlrm_s_pytorch import unpack_batch
def cond_features(dlrm, xloader, network, train_mode=False, num_batch=5, use_gpu=True, ndevices=1):
    cond_features = 0
    device = torch.cuda.current_device()
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
    
    smallest, largest = features_get_singular_values(logit)
    return largest - smallest