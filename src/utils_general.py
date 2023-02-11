import matplotlib.pyplot as plt
import numpy as np
import torch

from .utils_model import jit_batch_spec_to_Sqt
from .utils_convolution import interp_nb, get_I_conv

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