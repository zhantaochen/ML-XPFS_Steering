import torch
import torch.nn.functional as F
from math import pi
import numpy as np

def get_kernel_by_sigma(sigma, step):
    boundary = torch.ceil(3 * sigma)
    x_kernel = torch.arange(-boundary.item(), boundary.item()+step, step)
    kernel = torch.exp(-x_kernel.pow(2)/(2*sigma.pow(2))) / (sigma * np.sqrt(2*pi))
    return kernel / kernel.sum()

def lorentzian(x, x0, a, gam):
    x = x[None].repeat_interleave(len(x0), dim=0)
    y = torch.abs(a) * gam**2 / ( gam**2 + ( x - x0 )**2)
    return y.sum(dim=0)

def voigt(x, x0, a, gam, sig):
    step = x[1] - x[0]
    x = x[None].repeat_interleave(len(x0), dim=0)
    y = torch.abs(a) * gam**2 / ( gam**2 + ( x - x0 )**2)
    kernel = []
    for _s in sig:
        kernel.append(get_kernel_by_sigma(_s, step.item()))
    out = []
    for i in range(len(x0)):
        out.append(
            F.conv1d(y[i].squeeze().unsqueeze(0).unsqueeze(0), kernel[i].squeeze().unsqueeze(0).unsqueeze(0).double(), padding='same')
        )
    return torch.vstack(out).sum(dim=0).squeeze()

def fit_peak(xData, yData, numIters=1000, mode='lorentzian'):
    yData = 10 * yData / yData.max()

    param_X0 = torch.nn.Parameter(torch.tensor([[10.], [13.]]))
    param_A = torch.nn.Parameter(torch.tensor([[yData.max()], [yData.max()]]))
    param_G = torch.nn.Parameter(torch.tensor([[0.5], [0.5]]))
    if mode == 'lorentzian':
        opt = torch.optim.Adam([param_X0, param_A, param_G], lr=0.1)
    elif mode == 'voigt':
        param_sigma = torch.nn.Parameter(torch.tensor([[0.2], [0.2]]))
        opt = torch.optim.Adam([param_X0, param_A, param_G, param_sigma], lr=0.1)

    for i in range(numIters):
        if mode == 'lorentzian':
            yFit = lorentzian(xData, param_X0, param_A, param_G)
        elif mode == 'voigt':
            yFit = voigt(xData, param_X0, param_A, param_G, param_sigma)
        loss = F.mse_loss(yFit, yData)

        print("loss = ", loss.item(), end='\r')
        opt.zero_grad()
        loss.backward()
        opt.step()
    if mode == 'lorentzian':
        return param_X0.detach(), param_A.detach(), param_G.detach()
    elif mode == 'voigt':
        return param_X0.detach(), param_A.detach(), param_G.detach(), param_sigma.detach()