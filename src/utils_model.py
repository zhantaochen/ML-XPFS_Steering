import torch
import numpy as np
import scipy.constants as const

meV_to_2piTHz = 2 * np.pi * 1e-15 / const.physical_constants['hertz-electron volt relationship'][0]

def fc_block(feat_in, feat_out, bias=True, nonlin="relu", batch_norm=False):
    modules = [torch.nn.Linear(feat_in, feat_out, bias=bias)]
    if batch_norm:
        modules.append(torch.nn.BatchNorm1d(feat_out))
    if nonlin == "relu":
        modules.append(torch.nn.ReLU())
    elif nonlin is None:
        pass
    return modules

def construct_fc_net(feat_in, feat_out, feat_hid_list, fc_kwargs={}):
    if feat_hid_list is None:
        feat_hid_list = [256, 64, 16]
    fc = [*fc_block(feat_in, feat_hid_list[0], **fc_kwargs)]
    for i, (feat_hid_1, feat_hid_2) in enumerate(
        zip(feat_hid_list[:-1], feat_hid_list[1:])
    ):
        fc += fc_block(feat_hid_1, feat_hid_2, **fc_kwargs)
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


def lorentzian(center, Gamma, intensity, resolution=0.1, minimum=None):
    if minimum is not None:
        w = torch.arange(max(0, center-8*Gamma), center+8*Gamma+resolution, resolution)
    else:
        w = torch.arange(center-8*Gamma, center+8*Gamma+resolution, resolution)
    l = intensity/np.pi * 0.5*Gamma / ((w-center)**2 + (0.5*Gamma)**2)
    return w, l