import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .utils_model import jit_batch_spec_to_Sqt, array2tensor, tensor2array
from .utils_convolution import interp_nb, get_I_conv

@torch.jit.script
def get_I(t, y, gamma, pulse_width, meV_to_2piTHz, 
          elas_amp=torch.tensor([0.0]), elas_wid=torch.tensor([1.0])):
    n_particles = gamma.shape[0]
    t = t.to(y)
    elas_amp = torch.atleast_2d(elas_amp).to(y)
    elas_wid = torch.atleast_2d(elas_wid).to(y)
    if elas_amp.shape[0] == 1:
        elas_amp = elas_amp.repeat_interleave(n_particles, dim=0)
    if elas_wid.shape[0] == 1:
        elas_wid = elas_wid.repeat_interleave(n_particles, dim=0)
    
    omega, inten = torch.split(y, (y.shape[1]//2, y.shape[1]//2), dim=1)
    omega_full = torch.arange(0, 25, 0.5).to(y).unsqueeze(0).repeat_interleave(n_particles, dim=0)
    inten_elas = (elas_amp * torch.exp(-omega_full.pow(2)/(2 * elas_wid**2))).to(y)
    # Sqt_bkg = jit_batch_spec_to_Sqt(omega_full, inten_elas, t, meV_to_2piTHz).sum(dim=1).squeeze()

    if pulse_width > 0.:
        ## method 1
        # t = t.to(y)
        step = 10
        step_size = pulse_width / step
        t_extended = torch.vstack([t - pulse_width/2 + n * step_size for n in range(step+1)]).T.flatten().to(y)
        ## method 2
        # t_extended = [torch.linspace((_t-pulse_width/2).item(), (_t+pulse_width/2).item(), int(pulse_width/0.01)).to(y) for _t in t]
        # t_extended = [torch.linspace((_t-pulse_width/2).item(), (_t+pulse_width/2).item(), 11).to(y) for _t in t]
        # t_extended = torch.vstack(t_extended).flatten().to(y)
        ## method 3
        # step = 11
        # delta_t = torch.linspace(-pulse_width/2, pulse_width/2, step+1)
        # t_extended = (t[:,None] + delta_t[None,:].to(t)).flatten()

        Sqt_envelope = torch.exp(-torch.einsum("bm,nt->bmnt", F.relu(gamma), t_extended.abs().reshape(len(t),-1)))
        Sqt_mag = jit_batch_spec_to_Sqt(
            omega, inten, t_extended, meV_to_2piTHz
            ).sum(dim=1, keepdim=True).reshape(omega.shape[0],1,len(t),-1) * Sqt_envelope
        Sqt_bkg = jit_batch_spec_to_Sqt(
            omega_full, inten_elas, t_extended, meV_to_2piTHz
            ).sum(dim=1, keepdim=True).reshape(omega.shape[0],1,len(t),-1)
        I_pred = torch.trapz(
            (Sqt_mag + Sqt_bkg).abs().pow(2), 
                t_extended.reshape(len(t),-1), dim=-1) / pulse_width
    else:
        # t = t.to(y)
        S_envelope = torch.exp(-torch.einsum("bm,nt->bmnt", F.relu(gamma), t.abs().reshape(len(t),-1)))
        Sqt_mag = jit_batch_spec_to_Sqt(
            omega, inten, t, meV_to_2piTHz
            ).sum(dim=1, keepdim=True).reshape(omega.shape[0],1,len(t),-1) * S_envelope
        Sqt_bkg = jit_batch_spec_to_Sqt(
            omega_full, inten_elas, t, meV_to_2piTHz
            ).sum(dim=1, keepdim=True).reshape(omega.shape[0],1,len(t),-1)
        I_pred = (Sqt_mag + Sqt_bkg).abs().pow(2)
    return I_pred

def prepare_sample(x, y, gamma, times, 
                   pulse_width=0.1, normalize_to_value=None,
                   elas_amp_factor=0., elas_wid=1.0, elas_amp_abs_max=None, visualize=False):

    # prepare Sqt energies and intensities
    omega_test, inten_test = torch.split(y, y.shape[0]//2)
    omega_full = torch.arange(0, omega_test.max() + 5 + 10 * gamma, 0.01)
    _omega_full = omega_full[None].repeat_interleave(len(omega_test), dim=0)
    inten_full = (torch.abs(inten_test)[:,None] * gamma**2 / (gamma**2 + (_omega_full - omega_test[:,None])**2)).sum(dim=0)

    # setup time for Sqt computation
    dt = times[1]-times[0]
    times_extended = np.arange(times[0]-pulse_width, times[-1]+pulse_width, dt)
    times_extended_tensor = torch.from_numpy(times_extended)

    # prepare elastic scattering intensities
    if elas_amp_abs_max is not None:
        elas_amp = min(elas_amp_factor * inten_test.max(), elas_amp_abs_max)
    else:
        elas_amp = elas_amp_factor * inten_test.max()
    true_pars = x.cpu().numpy().tolist() + [gamma, elas_amp, elas_wid]
    inten_elas = elas_amp * torch.exp(-omega_full.pow(2)/(2 * elas_wid**2))
    Sqt_bkg = jit_batch_spec_to_Sqt(
        omega_full, inten_elas, times_extended_tensor).sum(dim=1).squeeze()

    # S and |S^2| with NO pulse shape convolution
    # Sqt_mag = jit_batch_spec_to_Sqt(omega_test, inten_test, times_extended_tensor).sum(dim=1).squeeze() * \
    #     torch.exp(- gamma * times_extended_tensor)
    Sqt_mag = jit_batch_spec_to_Sqt(omega_full, inten_full, times_extended_tensor).sum(dim=1).squeeze()
    true_S = (Sqt_mag + Sqt_bkg).detach().cpu().numpy()
    if normalize_to_value is not None:
        S_norm_factor = np.sqrt(normalize_to_value) / true_S[int(pulse_width / dt)]
    else:
        S_norm_factor = 1.
    true_S = S_norm_factor * true_S
    func_I_noconv = lambda t: interp_nb(t, times_extended, np.abs(true_S)**2)

    # S and |S^2| with pulse shape convolution
    true_I_conv = get_I_conv(times, times_extended, true_S, pulse_width)
    if normalize_to_value is not None:
        I_norm_factor = normalize_to_value / true_I_conv[0]
    else:
        I_norm_factor = 1.
    true_I_conv = I_norm_factor * true_I_conv
    func_I_conv = lambda t: interp_nb(t, times, true_I_conv)

    if visualize:
        fig, ax = plt.subplots(1,1)
        ax.plot(omega_full, inten_elas)
        ax.plot(omega_full, inten_full)

        fig, ax = plt.subplots(1,1)
        ax.plot(times_extended, Sqt_mag * S_norm_factor)
        ax.plot(times_extended, Sqt_bkg * S_norm_factor)
        ax.plot(times_extended, S_norm_factor * jit_batch_spec_to_Sqt(omega_full, inten_full+inten_elas, times_extended_tensor).sum(dim=1).squeeze())
        ax.plot(times_extended, true_S)
        # ax.plot(times, true_I_conv)

        fig, ax = plt.subplots(1,1)
        ax.plot(times_extended, np.abs(true_S)**2)
        ax.plot(times, true_I_conv)

    return np.asarray(true_pars), func_I_conv, func_I_noconv