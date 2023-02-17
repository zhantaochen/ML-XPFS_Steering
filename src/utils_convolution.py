from numba import njit, prange
import numpy as np

@njit
def interp_nb(t, tp, Sp):
    return np.interp(t, tp, Sp)

@njit
def get_I_conv_at_t(t, tp, Sp, pulse_width):
    if pulse_width > 0.0:
        t_around = np.linspace(t-pulse_width/2, t+pulse_width/2, int(pulse_width/0.01))
        I_out = np.trapz(np.abs(interp_nb(t_around, tp, Sp))**2, t_around) / pulse_width
    else:
        I_out = np.abs(interp_nb(t, tp, Sp))**2
    return I_out

@njit(parallel=True)
def get_I_conv(t, tp, Sp, pulse_width):
    if isinstance(t, (int, float)):
        t = np.asarray([t,])
    I_out = np.zeros_like(t)
    for i_t in prange(len(t)):
        I_out[i_t] = get_I_conv_at_t(t[i_t], tp, Sp, pulse_width)
    # I_0 = get_I_conv_at_t(func_S, 0., pulse_width)
    # I_out = I_out / I_0
    return I_out