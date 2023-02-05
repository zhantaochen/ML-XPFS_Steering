import torch
import numpy as np
import scipy.constants as const
from scipy.interpolate import interp1d

meV_to_2piTHz = 2 * np.pi * 1e-15 / const.physical_constants['hertz-electron volt relationship'][0]
meV_to_2piTHz_tensor = torch.tensor(2 * np.pi * 1e-15 / const.physical_constants['hertz-electron volt relationship'][0])

def fc_block(feat_in, feat_out, bias=True, nonlin="relu", batch_norm=False):
    modules = [torch.nn.Linear(feat_in, feat_out, bias=bias)]
    if batch_norm:
        modules.append(torch.nn.BatchNorm1d(feat_out))
    if nonlin == "relu":
        modules.append(torch.nn.ReLU())
    elif nonlin is None:
        pass
    return modules

def construct_fc_net(feat_in, feat_out, feat_hid_list, fc_kwargs={}, act_last=False, dropout_rate=None):
    if feat_hid_list is None:
        feat_hid_list = [256, 128, 64]
    fc = [*fc_block(feat_in, feat_hid_list[0], **fc_kwargs)]
    if dropout_rate is not None:
        fc += [torch.nn.Dropout(p=dropout_rate, inplace=False)]
    for i, (feat_hid_1, feat_hid_2) in enumerate(
        zip(feat_hid_list[:-1], feat_hid_list[1:])
    ):
        fc += fc_block(feat_hid_1, feat_hid_2, **fc_kwargs)
        if dropout_rate is not None:
            fc += [torch.nn.Dropout(p=dropout_rate, inplace=False)]
    if act_last:
        fc += fc_block(feat_hid_list[-1], feat_out, bias=False, nonlin='relu', batch_norm=False)
    else:
        fc += fc_block(feat_hid_list[-1], feat_out, bias=False, nonlin=None, batch_norm=False)
    fc = torch.nn.Sequential(*fc)
    return fc

def spec_to_Sqt(omega, inten, time, keepmod=False):
    S = 0.
    for _inten, _omega in zip(inten, omega):
        _inten = torch.atleast_1d(_inten)
        _omega = torch.atleast_1d(_omega)
        _S_tmp = (_inten[:,None] * torch.cos(meV_to_2piTHz * torch.einsum("w, t -> wt", _omega, time)))
        if keepmod:
            S = S + _S_tmp
        else:
            S = S + _S_tmp.sum(dim=0)
    return S

def batch_spec_to_Sqt(omega, inten, time):
    inten = torch.atleast_2d(inten)
    omega = torch.atleast_2d(omega)
    return torch.einsum("bm, bmt -> bmt", inten, torch.cos(meV_to_2piTHz * torch.einsum("bw, t -> bwt", omega, time)))


@torch.jit.script
def jit_batch_spec_to_Sqt(omega, inten, time, meV_to_2piTHz=meV_to_2piTHz_tensor):
    inten = torch.atleast_2d(inten)
    omega = torch.atleast_2d(omega)
    return torch.einsum("bm, bmt -> bmt", inten, torch.cos(meV_to_2piTHz * torch.einsum("bw, t -> bwt", omega, time)))
    # return torch.einsum("bm, bmt -> bmt", inten, torch.cos(meV_to_2piTHz * omega[...,None] * time[None,None,:]))

def lorentzian(center, Gamma, intensity, resolution=0.1, minimum=None):
    if minimum is not None:
        w = torch.arange(max(0, center-8*Gamma), center+8*Gamma+resolution, resolution)
    else:
        w = torch.arange(center-8*Gamma, center+8*Gamma+resolution, resolution)
    l = 1 /np.pi * 0.5*Gamma / ((w-center)**2 + (0.5*Gamma)**2)
    l = intensity * l / torch.trapz(l, w)
    return w, l

def lorentzian_on_grid(center, Gamma, intensity, w_grid):
    w_grid_large = torch.arange(
        min(w_grid.min(), center-10*Gamma), 
        max(w_grid.max(), center+10*Gamma), 
        w_grid[1] - w_grid[0]
        )
    l = 1 /np.pi * 0.5*Gamma / ((w_grid-center)**2 + (0.5*Gamma)**2)
    l =  intensity * l / torch.trapz(l, w_grid)
    return l

def array2tensor(arr):
    if isinstance(arr, np.ndarray):
        tensor = torch.from_numpy(arr)
        return tensor
    elif isinstance(arr, torch.Tensor):
        return arr
    else:
        raise ValueError("Input should be either numpy array or torch tensor.")

def tensor2array(tensor):
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
        return arr
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise ValueError("Input should be either numpy array or torch tensor.")