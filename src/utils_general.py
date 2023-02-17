import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .utils_model import jit_batch_spec_to_Sqt, array2tensor, tensor2array
from .utils_convolution import interp_nb, get_I_conv

@torch.jit.script
def get_I(t, y, gamma, pulse_width, meV_to_2piTHz):
    omega, inten = torch.split(y, (y.shape[1]//2, y.shape[1]//2), dim=1)
    if pulse_width > 0.:
        t_extended = [torch.linspace((_t-pulse_width/2).item(), (_t+pulse_width/2).item(), int(pulse_width/0.01)).to(y) for _t in t]
        t_extended = torch.vstack(t_extended).flatten().to(y)
        S_envelope = torch.exp(-torch.einsum("bm,nt->bmnt", F.relu(gamma), t_extended.abs().reshape(len(t),-1)))
        I_pred = torch.trapz(
            (jit_batch_spec_to_Sqt(
                omega, inten, t_extended, meV_to_2piTHz
                ).sum(dim=1, keepdim=True).reshape(omega.shape[0],1,len(t),-1) * S_envelope).abs().pow(2), 
                t_extended.reshape(len(t),-1), dim=-1) / pulse_width
    else:
        # t = array2tensor(t)
        t = t.to(y)
        # print(t.device, y.device, gamma.device, pulse_width.device, meV_to_2piTHz.device)
        S_envelope = torch.exp(-torch.einsum("bm,nt->bmnt", F.relu(gamma), t.abs().reshape(len(t),-1)))
        I_pred = (jit_batch_spec_to_Sqt(
                        omega, inten, t, meV_to_2piTHz
                    ).sum(dim=1, keepdim=True).reshape(omega.shape[0],1,len(t),-1) * S_envelope).abs().pow(2)
    return I_pred

def prepare_sample(x, y, gamma, times, pulse_width=0.1, visualize=False, normalize_to_value=None):
    # prepare Sqt energies and intensities
    omega_test, inten_test = torch.split(y, y.shape[0]//2)
    true_pars = x.cpu().numpy().tolist() + [gamma,]

    # setup time for Sqt computation
    dt = times[1]-times[0]
    times_extended = np.arange(times[0]-pulse_width, times[-1]+pulse_width, dt)
    times_extended_tensor = torch.from_numpy(times_extended)

    # S and |S^2| with NO pulse shape convolution
    true_S = jit_batch_spec_to_Sqt(omega_test, inten_test, times_extended_tensor).sum(dim=1).squeeze() * \
        torch.exp(- gamma * times_extended_tensor)
    true_S = true_S.detach().cpu().numpy()
    if normalize_to_value is not None:
        true_S = np.sqrt(normalize_to_value) * true_S / true_S[int(pulse_width / dt)]
    func_I_noconv = lambda t: interp_nb(t, times_extended, np.abs(true_S)**2)

    # S and |S^2| with pulse shape convolution
    true_I_conv = get_I_conv(times, times_extended, true_S, pulse_width)
    if normalize_to_value is not None:
        true_I_conv = normalize_to_value * true_I_conv / true_I_conv[0]
    func_I_conv = lambda t: interp_nb(t, times, true_I_conv)

    if visualize:
        fig, ax = plt.subplots(1,1)
        ax.plot(times_extended, np.abs(true_S)**2)
        ax.plot(times, true_I_conv)

    return np.asarray(true_pars), func_I_conv, func_I_noconv