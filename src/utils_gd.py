from collections import namedtuple
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .utils_model import array2tensor, tensor2array, jit_batch_spec_to_Sqt
from .utils_general import get_I
import numpy as np
import scipy.constants as const
meV_to_2piTHz = torch.tensor(2 * np.pi * 1e-15 / const.physical_constants['hertz-electron volt relationship'][0])

def get_I_from_params(model, t, parameters, pulse_width, norm_I0=100, model_uncertainty=False, device='cpu'):
    model.to(device)
    t = t.to(device)
    for k_p in parameters.keys():
        parameters[k_p] = parameters[k_p].to(device)
    x = torch.cat((parameters.J, parameters.D), dim=1)
    if not model_uncertainty:
        y = model(x.to(device))
    else:
        y_mu, y_var = model(x.to(device))
        y = torch.distributions.MultivariateNormal(y_mu, y_var).sample()
    pulse_width = torch.tensor(pulse_width).to(y).clone().detach()
    meV_to_2piTHz = torch.tensor(2 * np.pi * 1e-15 / const.physical_constants['hertz-electron volt relationship'][0])
    I_pred = get_I(t, y, parameters.gamma, pulse_width, meV_to_2piTHz)
    I_pred_t0 = get_I(torch.tensor([0,]), y, parameters.gamma, pulse_width, meV_to_2piTHz)
    I_out = I_pred / I_pred_t0 * norm_I0
    return I_out

def fit_measurement_with_OptBayesExpt_parameters(
    model, t, S, params, pulse_width, norm_I0=100,
    batch_size=10, maxiter=100, lr=0.001,
    params_type='mean_std_normal', save_param_hist=True, verbose=False, model_uncertainty=False, device='cpu'
):
    """_summary_

    Parameters
    ----------
    t : _type_
        _description_
    S : _type_
        _description_
    batch_size : int, optional
        _description_, by default 10
    maxiter : int, optional
        _description_, by default 100
    lr : float, optional
        _description_, by default 0.001
    retrain_criteria : tuple (N, M), optional
        if loss does not decrease by M in N steps, perform new training, by default None

    Returns
    -------
    _type_
        _description_
    """
    model.to(device)
    t = torch.atleast_1d(array2tensor(t)).to(device)
    S = torch.atleast_1d(array2tensor(S)).to(device)

    parameters = torch.nn.ParameterDict()
    for (name, a, b) in zip(*params):
        if params_type == 'mean_std_normal':
            parameters[name] = (a + b*torch.randn(batch_size,1)).requires_grad_(True)
        elif params_type == 'min_max_uniform':
            parameters[name] = (a + (b-a)*torch.rand(batch_size,1)).requires_grad_(True)
        elif params_type == 'particles':
            parameters[name] = array2tensor(a).unsqueeze(1).requires_grad_(True)
    parameters = parameters.to(device)
    param_lst = []
    for name in params[0]:
        param_lst.append({'params': eval(f'parameters.{name}')})
    optimizer = torch.optim.Adam(param_lst, lr=lr)

    loss_hist = []
    if save_param_hist: 
        param_hist = {name: [] for name in params[0]}

    if verbose:
        pbar = tqdm(range(maxiter))
    else:
        pbar = range(maxiter)

    for i_iter in pbar:
        # x = torch.cat((self.J, self.D, self.K), dim=1)
        with torch.no_grad():
            parameters.gamma.data = parameters.gamma.data.abs()
        x = torch.cat((parameters.J, parameters.D), dim=1)
        if not model_uncertainty:
            y = model(x.to(device))
        else:
            y_mu, y_var = model(x.to(device))
            y = torch.distributions.MultivariateNormal(y_mu, y_var).sample()
        # omega, inten = torch.split(y, [y.shape[1]//2, y.shape[1]//2], dim=1)
        # # batch x mode x time
        # S_envelope = torch.exp(-torch.einsum("bm,t->bmt", F.relu(parameters.gamma), t.to(device)))
        # S_pred = (jit_batch_spec_to_Sqt(omega, inten, t) * S_envelope).sum(dim=1)
        # S_pred = torch.abs(S_pred)**2
        # S_pred = S_pred / S_pred[:,0,None] * S[0]
        pulse_width = torch.tensor(pulse_width).to(y).clone().detach()
        meV_to_2piTHz = torch.tensor(2 * np.pi * 1e-15 / const.physical_constants['hertz-electron volt relationship'][0])
        I_pred = get_I(t, y, parameters.gamma, pulse_width, meV_to_2piTHz, parameters.elas_amp, parameters.elas_wid)
        I_pred_t0 = get_I(torch.tensor([0,]), y, parameters.gamma, pulse_width, meV_to_2piTHz, parameters.elas_amp, parameters.elas_wid)
        I_out = I_pred / I_pred_t0 * norm_I0

        loss_batch = (I_out.squeeze() - torch.atleast_2d(S).repeat_interleave(batch_size,dim=0).to(I_out)).pow(2).mean(dim=1)
        loss = loss_batch.mean()
        
        # J_old = parameters.J.data.clone()
        # D_old = parameters.D.data.clone()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # J_new = parameters.J.data.clone()
        # D_new = parameters.D.data.clone()
        # print("J change: ", (J_new - J_old).abs().mean())
        # print("D change: ", (D_new - D_old).abs().mean())

        loss_hist.append(loss.item())
        
        # if replace_worst_with_mean and ((loss_batch.max().abs() - loss_batch.min().abs())/loss_batch.min().abs() > 5.0):
        #     idx_loss_descending = torch.argsort(loss_batch, descending=True)
        #     idx_worst = idx_loss_descending[:2]
        #     idx_best =  idx_loss_descending[-2:]
            # with torch.no_grad():
            #     self.gamma.data[idx_worst] = self.gamma.data[idx_best].mean(dim=0)
            #     self.J.data[idx_worst] = self.J.data[idx_best].mean() + torch.randn_like(self.J.data[idx_worst]) * 0.01
            #     self.D.data[idx_worst] = self.D.data[idx_best].mean() + torch.randn_like(self.J.data[idx_worst]) * 0.01
            #     self.K.data[idx_worst] = self.K.data[idx_best].mean() + torch.randn_like(self.J.data[idx_worst]) * 0.01
        # print(self.J)
        if save_param_hist: 
            for name in params[0]:
                param_hist[name].append(eval(f'parameters.{name}.clone().detach().cpu()'))
    for key in param_hist.keys():
        param_hist[key] = torch.cat(param_hist[key], dim=-1).T
    return loss_hist, param_hist