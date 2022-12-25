import optbayesexpt as obe
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .utils_model import construct_fc_net, array2tensor, tensor2array

import scipy.constants as const
meV_to_2piTHz = torch.tensor(2 * np.pi * 1e-15 / const.physical_constants['hertz-electron volt relationship'][0])

@torch.jit.script
def jit_batch_spec_to_Sqt(omega, inten, time, meV_to_2piTHz=meV_to_2piTHz):
    inten = torch.atleast_2d(inten)
    omega = torch.atleast_2d(omega)
    return torch.einsum("bm, bmt -> bmt", inten, torch.cos(meV_to_2piTHz * torch.einsum("bw, t -> bwt", omega, time)))
    
class OptBayesExpt_CustomCost(obe.OptBayesExpt):
    def __init__(self, cost_repulse_height=0.5, cost_repulse_width=0.05, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_repulse_width = cost_repulse_width
        self.cost_repulse_height = cost_repulse_height
        self.reset_proposed_setting()

    def cost_estimate(self):
        # if not hasattr(self, 'proposed_settings'):
        #     return 1.0
        # else:
        bins = self.proposed_settings["setting_bin"]
        cost = np.ones_like(self.setting_indices).astype('float')
        for idx in np.nonzero(bins)[0]:
            cost += self.cost_repulse_height * bins[idx] * np.squeeze(np.exp(-((self.allsettings - self.allsettings[:,idx]) / self.cost_repulse_width)**2))
        # return 1.0
        return cost


class BayesianInference:
    NUM_SAMPLES_PER_STEP = 10
    meV_to_2piTHz = torch.tensor(2 * np.pi * 1e-15 / const.physical_constants['hertz-electron volt relationship'][0])
    def __init__(self, model, settings=(), parameters=(), constants=(), pulse_width=None, batch_size=500, scale_factor=1.):
        """_summary_

        Parameters
        ----------
        model : torch.nn.Module
        param_prior : dict
            {'p1': {'mu': 0., 'sigma': 1.}}
        settings : tuple/list of arrays/tensors
        """
        # super().__init__()
        # self.register_module('forward_model', model)
        self.forward_model = model
        self.settings = settings
        self.parameters = parameters
        self.constants = constants
        self.pulse_width = pulse_width

        if self.pulse_width is not None:
            self.model_function = self.model_function_conv
        else:
            self.model_function = self.model_function_noconv
        self.init_OptBayesExpt()
        self.init_saving_lists()

    def init_OptBayesExpt(self, ):
        self.obe_model = OptBayesExpt_CustomCost(
            measurement_model=self.model_function, setting_values=self.settings, parameter_samples=self.parameters, 
            constants=self.constants, cost_repulse_height=0.25, cost_repulse_width=0.05)
        # self.obe_model = obe.OptBayesExpt(measurement_model=self.model_function, setting_values=self.settings, parameter_samples=self.parameters, constants=self.constants)
        
    def init_saving_lists(self, ):
        self.measured_settings = []
        self.measured_observables = []
        self.param_mean = []
        self.param_std = []
        self.utility_list = []
        self.model_predictions_on_obe_mean = []

    def model_function_noconv(self, sets, pars, cons):
        """ Evaluates a trusted model of the experiment's output
        The equivalent of a fit function. The argument structure is
        required by OptBayesExpt.
        Args:
            sets: A tuple of setting values, or a tuple of settings arrays
            pars: A tuple of parameter arrays or a tuple of parameter values
            cons: A tuple of floats
        Returns:  the evaluated function
        """
        # unpack the settings
        # t, = sets
        # unpack model parameters
        t, = sets
        J, D, gamma = pars
        if isinstance(t, (int, float)):
            t = torch.tensor([t,])
        else:
            t = torch.atleast_1d(array2tensor(t))
        if isinstance(gamma, (int, float)):
            gamma = torch.atleast_2d(torch.tensor([gamma]))
        else:
            gamma = array2tensor(gamma)[:,None]

        if isinstance(J, (int, float)):
            J = torch.tensor([[J]])
            D = torch.tensor([[D]])
        else:
            J = array2tensor(J)[:,None]
            D = array2tensor(D)[:,None]

        x = torch.cat((J, D), dim=1)
        y = self.forward_model(x.to(self.forward_model.device)).cpu()

        omega, inten = torch.split(y, (y.shape[1]//2, y.shape[1]//2), dim=1)
        S_envelope = torch.exp(-torch.einsum("bm,t->bmt", F.relu(gamma), t))
        S_pred = (jit_batch_spec_to_Sqt(omega, inten, t, self.meV_to_2piTHz) * S_envelope).sum(dim=1)
        I_pred = torch.abs(S_pred)**2
        # S_pred = S_pred / batch_spec_to_Sqt(omega, inten, torch.tensor([0,])).sum(dim=1)
        # calculate the Lorentzian
        return I_pred.detach().cpu().squeeze().numpy()

    def model_function_conv(self, sets, pars, cons):
        """ Evaluates a trusted model of the experiment's output
        The equivalent of a fit function. The argument structure is
        required by OptBayesExpt.
        Args:
            sets: A tuple of setting values, or a tuple of settings arrays
            pars: A tuple of parameter arrays or a tuple of parameter values
            cons: A tuple of floats
        Returns:  the evaluated function
        """
        # unpack the settings
        # t, = sets
        # unpack model parameters
        t, = sets
        J, D, gamma = pars
        if isinstance(t, (int, float)):
            t = torch.tensor([t,])
        else:
            t = torch.atleast_1d(array2tensor(t))
        if isinstance(gamma, (int, float)):
            gamma = torch.atleast_2d(torch.tensor([gamma]))
        else:
            gamma = array2tensor(gamma)[:,None]

        if isinstance(J, (int, float)):
            J = torch.tensor([[J]])
            D = torch.tensor([[D]])
        else:
            J = array2tensor(J)[:,None]
            D = array2tensor(D)[:,None]

        x = torch.cat((J, D), dim=1)
        y = self.forward_model(x.to(self.forward_model.device)).cpu()

        # y = torch.tensor([[ 6.1967, 15.8520,  5.3372,  3.6628]])

        omega, inten = torch.split(y, (y.shape[1]//2, y.shape[1]//2), dim=1)
        t_extended = []
        # S_pred = torch.zeros((x.shape[0],1,t.shape[0]))
        for i_t, _t in enumerate(t):
            t_extended.append(
                torch.linspace(_t-self.pulse_width/2, _t+self.pulse_width/2, int(self.pulse_width/0.01)))
        t_extended = torch.vstack(t_extended).flatten()
        S_envelope = torch.exp(-torch.einsum("bm,nt->bmnt", F.relu(gamma), t_extended.abs().reshape(len(t),-1)))
        I_pred = torch.trapz(
            (jit_batch_spec_to_Sqt(
                omega, inten, t_extended, self.meV_to_2piTHz
                ).sum(dim=1, keepdim=True).reshape(omega.shape[0],1,len(t),-1) * S_envelope).abs().pow(2), 
                t_extended.reshape(len(t),-1), dim=-1) / self.pulse_width
        # S_pred = (batch_spec_to_Sqt(omega, inten, t) * S_envelope).sum(dim=1)
        # S_pred = S_pred / batch_spec_to_Sqt(omega, inten, torch.tensor([0,])).sum(dim=1)
        # calculate the Lorentzian
        return I_pred.detach().cpu().squeeze().numpy()

    def step_OptBayesExpt(self, func_measure, num_samples_per_step=10):
        
        # next_setting = self.obe_model.opt_setting()
        next_setting = self.obe_model.get_setting()
        if self.obe_model.selection_method in ['optimal', 'good']:
            self.utility_list.append(self.obe_model.utility_stored)
        else:
            self.utility_list.append(None)
        next_observable = func_measure.simdata(next_setting)
        
        measurement = (next_setting, next_observable, func_measure.noise_level)
        self.obe_model.pdf_update(measurement)

        self.measured_settings.append(next_setting)
        self.measured_observables.append(next_observable)
        self.param_mean.append(self.obe_model.mean())
        self.param_std.append(self.obe_model.std())
        # pars = [np.random.normal(self.obe_model.mean()[i], self.obe_model.std()[i], num_samples_per_step) for i in range(len(self.obe_model.mean()))]
        _, model_predictions_on_obe_mean = self.predict_all_settings()
        self.model_predictions_on_obe_mean.append(model_predictions_on_obe_mean)
            # if np.all(param_std[-1] < 1e-2):
            #     break

    def run_N_steps_OptBayesExpt(self, N, func_measure, ret_particles=False):
        print(f"using the {self.obe_model.selection_method} setting")
        if ret_particles:
            particles = [self.obe_model.particles.copy()]
            particle_weights = [self.obe_model.particle_weights.copy()]
        for i in tqdm(range(N), desc="Running OptBayesExpt"):
            self.step_OptBayesExpt(func_measure)
            if ret_particles:
                particles.append(self.obe_model.particles.copy())
                particle_weights.append(self.obe_model.particle_weights.copy())
        if ret_particles:
            return np.asarray(particles), np.asarray(particle_weights)
        
    def get_all_measurements(self,):
        settings = np.asarray(self.measured_settings)
        observables = np.asarray(self.measured_observables)
        return settings, observables

    def get_organized_measurements(self,):
        settings, observables = self.get_all_measurements()
        unique_settings  = []
        mean_observables = []
        std_observables  = []
        for setting in np.unique(settings):
            unique_settings.append(setting)
            idx = np.where(settings==setting)[0]
            mean_observables.append(observables[idx].mean())
            std_observables.append(observables[idx].std())
        unique_settings  = np.asarray(unique_settings)
        mean_observables = np.asarray(mean_observables)
        std_observables  = np.asarray(std_observables)
        return unique_settings, mean_observables, std_observables
    
    def run_gradient_desc_on_current_measurements(self, N, lr=0.01, init_bayes_guess=False, batch_size=10):
        unique_settings, mean_observables, std_observables = \
            self.get_organized_measurements()
        t = torch.from_numpy(unique_settings).squeeze()
        S = torch.from_numpy(mean_observables).squeeze()
        if init_bayes_guess:
            loss_hist, params_hist = self.forward_model.fit_measurement_with_OptBayesExpt_parameters(
                t, S, (('J', 'D', 'gamma'), self.obe_model.mean(), self.obe_model.std()), lr=lr,
                maxiter=N, batch_size=batch_size
            )
        else:
            loss_hist, params_hist = self.forward_model.fit_measurement_with_OptBayesExpt_parameters(
                t, S, (('J', 'D', 'gamma'), (-2.0, -0.5, 0.5), (0.5, 0.25, 0.25)), lr=lr,
                maxiter=N, batch_size=batch_size
            )
        return loss_hist, params_hist

    def predict_all_settings(self, parameters=None):
        if parameters is None:
            measurements = self.model_function(self.settings, self.obe_model.mean(), ())
        else:
            measurements = self.model_function(self.settings, parameters, ())
        return self.settings, measurements

    def measure_all_settings(self, func_measure):
        # settings = tensor2array(self.settings)
        measurements = func_measure.simdata(self.settings)
        return self.settings, measurements
    
    def update_OptBayesExpt_particles(self,):
        particles = torch.cat([eval(f'bayes.forward_model.{name}.data').T for name in ('J', 'D', 'gamma')], dim=0).cpu().numpy()
        self.obe_model.particles = particles
        self.obe_model.particle_weights = np.ones(self.obe_model.n_particles) / self.obe_model.n_particles