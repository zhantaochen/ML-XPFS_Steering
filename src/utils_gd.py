from collections import namedtuple
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .utils_model import array2tensor, tensor2array, jit_batch_spec_to_Sqt

def fit_measurement_with_OptBayesExpt_parameters(
    model, t, S, params,
    batch_size=10, maxiter=100, lr=0.001,
    save_param_hist=True, verbose=False, model_uncertainty=False, device='cpu'
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
    t = torch.atleast_1d(array2tensor(t)).to(device)
    S = torch.atleast_1d(array2tensor(S)).to(device)

    parameters = torch.nn.ParameterDict()
    for (name, mean, std) in zip(*params):
        parameters[name] = (mean + std*torch.randn(batch_size,1)).requires_grad_(True)
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
        x = torch.cat((parameters.J, parameters.D), dim=1)
        if not model_uncertainty:
            y = model(x.to(device))
        else:
            y_mu, y_var = model(x.to(device))
            y = torch.distributions.MultivariateNormal(y_mu, y_var).sample()
        omega, inten = torch.split(y, [y.shape[1]//2, y.shape[1]//2], dim=1)
        # batch x mode x time
        S_envelope = torch.exp(-torch.einsum("bm,t->bmt", F.relu(parameters.gamma), t.to(device)))
        S_pred = (jit_batch_spec_to_Sqt(omega, inten, t) * S_envelope).sum(dim=1)
        S_pred = torch.abs(S_pred)**2
        S_pred = S_pred / S_pred[:,0,None] * S[0]
        loss_batch = (S_pred - torch.atleast_2d(S).repeat_interleave(batch_size,dim=0).to(S_pred)).pow(2).mean(dim=1)
        loss = loss_batch.mean()
        # loss = F.mse_loss(S_pred, torch.atleast_2d(S).repeat_interleave(batch_size,dim=0).to(S_pred))
        
        if loss < 0.001 * S[0]:
            break
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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