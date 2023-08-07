import torch
import numpy as np
import matplotlib.pyplot as plt

from .load_spe import load_spe
import scipy.constants as const
from .fit_lorentzian import fit_peak, lorentzian, voigt
from ..utils_convolution import interp_nb
from ..utils_general import get_I, get_I_conv
from ..utils_model import jit_batch_spec_to_Sqt

# def prepare_CrI3_sample(times, pulse_width=0.2, std_increase_factor=1., amp_increase_factor=0., E_cutoff=5., N_samples=1000, S0_value=10, elas_bkg_amp=0., elas_bkg_wid=1.0):
#     fname = 'src/CrI3/fig3a1.spe'
#     HH0_raw, E_raw, Z_raw, Err_raw = load_spe(fname)
#     idx_K = np.argmin(np.abs(HH0_raw + 1/3))

#     E_K, Intens_K = E_raw[E_raw>E_cutoff], Z_raw[E_raw>E_cutoff,idx_K]

#     # param_X0, param_A, param_G = fit_lorentzian(torch.from_numpy(E_K), torch.from_numpy(Intens_K))
#     # Intens_K_fitted = lorentzian(torch.from_numpy(E_K), param_X0, param_A, param_G).detach().numpy()
#     param_X0, param_A, param_G, param_sigma = fit_peak(torch.from_numpy(E_K), torch.from_numpy(Intens_K), numIters=2000, mode='voigt')
#     Intens_K_fitted = voigt(torch.from_numpy(E_K), param_X0, param_A, param_G, param_sigma).detach().numpy()
#     plt.plot(E_K, Intens_K_fitted)
#     plt.plot(E_K, Intens_K / Intens_K.max() * Intens_K_fitted.max())

#     meV_to_2piTHz = 2 * np.pi * 1e-15 / const.physical_constants['hertz-electron volt relationship'][0]
#     dt = times[1]-times[0]
#     times_extended = np.arange(times[0]-pulse_width, times[-1]+pulse_width, dt)
#     Z_noisy = np.vstack([Intens_K_fitted[None],]*N_samples) + \
#         np.random.randn(N_samples, (E_raw>E_cutoff).sum()) * Err_raw[E_raw>E_cutoff, idx_K][None] * std_increase_factor
#     S_noisy = np.einsum(
#         "nw, wt, t -> nwt", 
#         Z_noisy, 
#         np.cos(meV_to_2piTHz * np.einsum("w, t -> wt", E_K, times_extended)), 
#         np.exp(amp_increase_factor * times_extended)).sum(axis=1)
#     S_noisy /= S_noisy.mean(axis=0)[int(pulse_width / dt)]
#     S_noisy *= S0_value
#     S_mean = S_noisy.mean(axis=0)
#     S_std = S_noisy.std(axis=0)


#     I_noisy = S_noisy ** 2
#     I_mean = I_noisy.mean(axis=0)
#     I_std = I_noisy.std(axis=0)
#     # get unconvolved interpolation function
#     func_I_noconv = lambda t: interp_nb(t, times_extended, np.abs(S_mean)**2)
#     # get convolved interpolation function
#     true_I_conv = get_I_conv(times, times_extended, S_mean, pulse_width)
#     true_I_conv = S0_value**2 * true_I_conv / true_I_conv[0]
#     func_I_conv = lambda t: interp_nb(t, times, true_I_conv)
#     fig, ax = plt.subplots(1,1)
#     ax.plot(times_extended, np.abs(S_mean)**2)
#     ax.plot(times, true_I_conv)
#     ax.fill_between(times_extended, I_mean-I_std, I_mean+I_std, alpha=0.25)

#     fig, axes = plt.subplots(1,2, sharey=True, gridspec_kw={'wspace': 0.0, 'width_ratios': [2, 1]})
#     ax = axes[0]
#     ax.imshow(np.log10(Z_raw), extent=[HH0_raw.min(), HH0_raw.max(), E_raw.min(), E_raw.max()], origin='lower')
#     ax.vlines(-1/3, E_raw.min(), E_raw.max(), linestyles='--', colors='k')
#     ax.set_aspect(2 * np.ptp(HH0_raw) / np.ptp(E_raw))
#     ax.set_xlabel('$h$ for $[h=h, k=h, l=0]$')
#     ax.set_ylabel('$E$ (eV)')
#     ax = axes[1]
#     ax.errorbar(Z_raw[E_raw>E_cutoff,idx_K], E_K, yerr=None, xerr=Err_raw[E_raw>E_cutoff, idx_K], color='k', ecolor='k')
#     ax.plot(Intens_K_fitted / Intens_K_fitted.max() * Intens_K.max(), E_K, color='b', linestyle='--')
#     ax.set_xticks([])
#     ax.set_xlabel('Intensities (a.u.)')
#     ax.set_ylim([0,20])

#     return None, func_I_conv, func_I_noconv


def prepare_CrI3_sample(times, pulse_width=0.2, 
                        amp_increase_factor=0., 
                        E_cutoff=5., S0_value=10, avg_Gamma=True,
                        elas_bkg_amp_factor=0., elas_bkg_wid=1.0, mode='voigt'):
    fname = 'src/CrI3/fig3a1.spe'
    HH0_raw, E_raw, Z_raw, Err_raw = load_spe(fname)
    idx_K = np.argmin(np.abs(HH0_raw + 1/3))

    E_K, Intens_K = E_raw[E_raw>E_cutoff], Z_raw[E_raw>E_cutoff,idx_K]
    E_full = torch.from_numpy(E_raw)

    # param_X0, param_A, param_G = fit_lorentzian(torch.from_numpy(E_K), torch.from_numpy(Intens_K))
    # Intens_K_fitted = lorentzian(torch.from_numpy(E_K), param_X0, param_A, param_G).detach().numpy()
    if mode == 'voigt':
        param_X0, param_A, param_G, param_sigma = fit_peak(
            torch.from_numpy(E_K), torch.from_numpy(Intens_K), 
            numIters=2000, mode='voigt')
        Intens_K_fitted = voigt(torch.from_numpy(E_K), param_X0, param_A, param_G, param_sigma).detach().numpy()
        Sqw_mag = voigt(E_full, param_X0, param_A, param_G, param_sigma).detach()
    elif mode == 'lorentzian':
        param_X0, param_A, param_G = fit_peak(
            torch.from_numpy(E_K), torch.from_numpy(Intens_K), 
            numIters=2000, mode='lorentzian')
        if avg_Gamma:
            param_G = torch.ones_like(param_G) * param_G.mean()
            print("mean gamma is: ", param_G)
        Intens_K_fitted = lorentzian(torch.from_numpy(E_K), param_X0, param_A, param_G).detach().numpy()
        # Sqw_mag = lorentzian(E_full, param_X0, param_A, param_G).detach()
        E_full_dense = torch.linspace(E_full.min(),E_full.max(),400)
        Sqw_mag = lorentzian(E_full_dense, param_X0, param_A, param_G).detach()
    
    fig, ax = plt.subplots(1,1)
    ax.plot(E_K, Intens_K_fitted)
    ax.plot(E_K, Intens_K / Intens_K.max() * Intens_K_fitted.max(), '--')

    # meV_to_2piTHz = 2 * np.pi * 1e-15 / const.physical_constants['hertz-electron volt relationship'][0]
    dt = times[1]-times[0]
    times_extended = np.arange(times[0]-pulse_width, times[-1]+pulse_width, dt)

    Sqt_mag = jit_batch_spec_to_Sqt(
        E_full_dense, Sqw_mag, torch.from_numpy(times_extended)).sum(dim=1).squeeze() \
        * torch.exp(amp_increase_factor * torch.from_numpy(times_extended).abs())

    Sqw_bkg = elas_bkg_amp_factor * Sqw_mag.max() * torch.exp(-E_full.pow(2)/(2 * elas_bkg_wid**2)).detach()
    Sqt_bkg = jit_batch_spec_to_Sqt(E_full, Sqw_bkg, torch.from_numpy(times_extended)).sum(dim=1).squeeze()

    Sqt_tot = (Sqt_mag + Sqt_bkg).numpy()
    norm_factor_Sqt = S0_value / Sqt_tot[int(pulse_width / dt)]
    Sqt_tot *= norm_factor_Sqt

    #debug
    Sqw_mag_debug = lorentzian(
        E_full_dense, param_X0, param_A, param_G-0.65).detach()
    Sqt_mag_debug = jit_batch_spec_to_Sqt(
        E_full_dense, Sqw_mag_debug, torch.from_numpy(times_extended)).sum(dim=1).squeeze() 
    Sqt_tot_debug = (Sqt_mag_debug + Sqt_bkg).numpy()
    norm_factor_Sqt_debug = S0_value / Sqt_tot_debug[int(pulse_width / dt)]
    Sqt_tot_debug *= norm_factor_Sqt_debug
    #debug

    fig, ax = plt.subplots(2,1)
    ax[0].plot(E_full_dense, Sqw_mag)
    ax[0].plot(E_full, Sqw_bkg)
    # ax[0].plot(E_full, Sqw_bkg+Sqw_mag)
    ax[1].plot(times_extended, Sqt_mag_debug.numpy()*norm_factor_Sqt_debug, '--')
    ax[1].plot(times_extended, Sqt_mag.numpy()*norm_factor_Sqt)
    ax[1].plot(times_extended, Sqt_bkg.numpy()*norm_factor_Sqt)
    # ax[1].plot(times_extended, Sqt_tot)

    # get unconvolved interpolation function
    func_I_noconv = lambda t: interp_nb(t, times_extended, np.abs(Sqt_tot)**2)
    # get convolved interpolation function
    true_I_conv = get_I_conv(times, times_extended, Sqt_tot, pulse_width)
    true_I_conv = S0_value**2 * true_I_conv / true_I_conv[0]
    func_I_conv = lambda t: interp_nb(t, times, true_I_conv)

    # fig, ax = plt.subplots(1,1)
    # ax.plot(times_extended, np.abs(Sqt_tot)**2)
    # ax.plot(times, true_I_conv)

    fig, axes = plt.subplots(1,2, sharey=True, gridspec_kw={'wspace': 0.0, 'width_ratios': [2, 1]})
    ax = axes[0]
    ax.imshow(np.log10(Z_raw), extent=[HH0_raw.min(), HH0_raw.max(), E_raw.min(), E_raw.max()], origin='lower')
    ax.vlines(-1/3, E_raw.min(), E_raw.max(), linestyles='--', colors='k')
    ax.set_aspect(2 * np.ptp(HH0_raw) / np.ptp(E_raw))
    ax.set_xlabel('$h$ for $[h=h, k=h, l=0]$')
    ax.set_ylabel('$E$ (eV)')
    ax = axes[1]
    ax.errorbar(Z_raw[E_raw>E_cutoff,idx_K], E_K, yerr=None, xerr=Err_raw[E_raw>E_cutoff, idx_K], color='k', ecolor='k')
    ax.plot(Intens_K_fitted / Intens_K_fitted.max() * Intens_K.max(), E_K, color='b', linestyle='--')
    ax.set_xticks([])
    ax.set_xlabel('Intensities (a.u.)')
    ax.set_ylim([0,20])

    return (None,None,param_G.mean().item()-amp_increase_factor,
            elas_bkg_amp_factor*Sqw_mag.max().item(),elas_bkg_wid), \
        func_I_conv, func_I_noconv