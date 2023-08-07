import torch
import torch.nn.functional as F

def lorentzian(x, x0, a, gam):
    x = x[None].repeat_interleave(len(x0), dim=0)
    y = torch.abs(a) * gam**2 / ( gam**2 + ( x - x0 )**2)
    return y.sum(dim=0)

def fit_lorentzian(xData, yData, numIters=1000):
    yData = 10 * yData / yData.max()

    param_X0 = torch.nn.Parameter(torch.tensor([[10.], [13.]]))
    param_A = torch.nn.Parameter(torch.tensor([[yData.max()], [yData.max()]]))
    param_G = torch.nn.Parameter(torch.tensor([[0.5], [0.5]]))
    opt = torch.optim.Adam([param_X0, param_A, param_G], lr=0.1)

    for i in range(numIters):
        loss = F.mse_loss(yData, lorentzian(xData, param_X0, param_A, param_G))

        print("loss = ", loss.item(), end='\r')
        opt.zero_grad()
        loss.backward()
        opt.step()
    return param_X0.detach(), param_A.detach(), param_G.detach()
