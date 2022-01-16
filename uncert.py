import numpy as np
import torch
from torch.autograd import grad
from Batch import batch_pad

def eval_hessian(loss_grad, model):
    cnt = 0
    for g in loss_grad:
        g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
        
        cnt = 1
    l = g_vector.size(0)
    hessian = torch.zeros(l, l)
    for idx in range(l):
        #for model in models:
        grad2rd = torch.autograd.grad(g_vector[idx], model.parameters(), create_graph=True)
        cnt = 0
        for g in grad2rd:
            
            g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
            cnt = 1
        hessian[idx] = g2
    return hessian


def evaluate_uncert(name, data_dict, eval_h):
    SSE = torch.nn.MSELoss(reduction='sum')
    SAE = torch.nn.L1Loss(reduction='sum')
    model = torch.load(name)
    scaling = model.scaling
    gmin = scaling['gmin']
    gmax = scaling['gmax']
    emin = scaling['emin']
    emax = scaling['emax']

    ids = np.array(list(data_dict.keys()))
    batch_info = batch_pad(data_dict,ids)
    b_fp = batch_info['b_fp']

    b_e_mask = batch_info['b_e_mask']
    b_fp.requires_grad = True
    sb_fp = (b_fp - gmin) / (gmax - gmin)
    N_atoms = batch_info['N_atoms'].view(-1)
    b_e = batch_info['b_e'].view(-1)
    b_f = batch_info['b_f'] 


    Atomic_Es = model(sb_fp)
    E_predict = torch.sum(Atomic_Es * b_e_mask, dim = [1,2])
    E_predict = E_predict/N_atoms
    E_predict = E_predict * (emax - emin) + emin

    sse = SSE(b_e, E_predict)

    rmse = torch.sqrt(sse/len(E_predict))

    if eval_h:
        loss_grad = torch.autograd.grad(sse, model.parameters(), create_graph=True, retain_graph=True)
        h = eval_hessian(loss_grad, model)
        return E_predict.detach().numpy(), b_e.detach().numpy(), rmse, h
    else:
        return E_predict.detach().numpy(), b_e.detach().numpy(), rmse

def get_fps(name,data_dict):
    model = torch.load(name)
    scaling = model.scaling
    gmin = scaling['gmin']
    gmax = scaling['gmax']
    emin = scaling['emin']
    emax = scaling['emax']

    ids = np.array(list(data_dict.keys()))
    batch_info = batch_pad(data_dict,ids)
    b_fp = batch_info['b_fp']
    b_e_mask = batch_info['b_e_mask']
    sb_fp = (b_fp - gmin) / (gmax - gmin)
    return b_fp.detach().numpy(), b_e_mask.detach().numpy()

