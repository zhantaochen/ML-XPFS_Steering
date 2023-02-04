from functools import partial
import optbayesexpt as obe
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .utils_model import construct_fc_net, array2tensor, tensor2array, batch_spec_to_Sqt

import scipy.constants as const
meV_to_2piTHz = torch.tensor(2 * np.pi * 1e-15 / const.physical_constants['hertz-electron volt relationship'][0])

@torch.jit.script
def jit_batch_spec_to_Sqt(omega, inten, time, meV_to_2piTHz=meV_to_2piTHz):
    inten = torch.atleast_2d(inten)
    omega = torch.atleast_2d(omega)
    return torch.einsum("bm, bmt -> bmt", inten, torch.cos(meV_to_2piTHz * torch.einsum("bw, t -> bwt", omega, time)))
    # return torch.einsum("bm, bmt -> bmt", inten, torch.cos(meV_to_2piTHz * omega[...,None] * time[None,None,:]))

class OptBayesExpt_CustomCost(obe.OptBayesExpt):
    def __init__(self, cost_repulse_height=0.5, cost_repulse_width=0.05, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_repulse_width = cost_repulse_width
        self.cost_repulse_height = cost_repulse_height
        self.reset_proposed_setting()

    def cost_estimate(self):
        bins = self.proposed_settings["setting_bin"]
        cost = np.ones_like(self.setting_indices).astype('float')
        for idx in np.nonzero(bins)[0]:
            cost += self.cost_repulse_height * bins[idx] * np.squeeze(np.exp(-((self.allsettings - self.allsettings[:,idx]) / self.cost_repulse_width)**2))
        return cost
        # return 1.0


class BayesianInference:
    NUM_SAMPLES_PER_STEP = 10
    meV_to_2piTHz = torch.tensor(2 * np.pi * 1e-15 / const.physical_constants['hertz-electron volt relationship'][0])
    def __init__(self, model, settings=(), parameters=(), constants=(), pulse_width=None, reference_setting_value=None):
        # super().__init__()
        # self.register_module('forward_model', model)
        self.forward_model = model
        self.settings = settings
        self.parameters = parameters
        self.constants = constants
        self.pulse_width = pulse_width

        if self.pulse_width is not None:
            self.model_function = partial(self.model_function_conv, ret_tensor=False)
            # self.model_function = self.model_function_conv_new
        else:
            self.model_function = partial(self.model_function_noconv, ret_tensor=False)
        self.init_OptBayesExpt()
        self.init_saving_lists()

        self.reference_setting_value = reference_setting_value

        # if device is None:
        #     self.__device = 'cuda' if torch.cuda.is_exist() else 'cpu'
        # else:
        #     self.__device = device

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

    def model_function_noconv(self, sets, pars, cons, ret_tensor=False):
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

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        t, = sets
        J, D, gamma = pars
        if isinstance(t, (int, float)):
            t = torch.tensor([t,])
        else:
            t = torch.atleast_1d(array2tensor(t))
        t = t.to(device)

        if isinstance(gamma, (int, float)):
            gamma = torch.atleast_2d(torch.tensor([gamma]))
        else:
            gamma = array2tensor(gamma)[:,None]
        gamma = gamma.to(device)

        if isinstance(J, (int, float)):
            J = torch.tensor([[J]])
            D = torch.tensor([[D]])
        else:
            J = array2tensor(J)[:,None]
            D = array2tensor(D)[:,None]

        self.forward_model.to(device)
        x = torch.cat((J, D), dim=1).to(device)
        y = self.forward_model(x)

        omega, inten = torch.split(y, (y.shape[1]//2, y.shape[1]//2), dim=1)
        S_envelope = torch.exp(-torch.einsum("bm,t->bmt", F.relu(gamma), t))
        S_pred = (jit_batch_spec_to_Sqt(omega, inten, t, self.meV_to_2piTHz) * S_envelope).sum(dim=1)
        I_pred = torch.abs(S_pred)**2
        # S_pred = S_pred / batch_spec_to_Sqt(omega, inten, torch.tensor([0,])).sum(dim=1)
        # calculate the Lorentzian
        if ret_tensor:
            return I_pred.squeeze()
        else:
            return I_pred.detach().cpu().squeeze().numpy()

    def model_function_conv(self, sets, pars, cons, ret_tensor=False):
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
        device = 'cpu'
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'

        t, = sets
        J, D, gamma = pars
        if isinstance(t, (int, float)):
            t = torch.tensor([t,])
        else:
            t = torch.atleast_1d(array2tensor(t))
        t = t.to(device)

        if isinstance(gamma, (int, float)):
            gamma = torch.atleast_2d(torch.tensor([gamma]))
        else:
            gamma = array2tensor(gamma)[:,None]
        gamma = gamma.to(device)

        if isinstance(J, (int, float)):
            J = torch.tensor([[J]])
            D = torch.tensor([[D]])
        else:
            J = array2tensor(J)[:,None]
            D = array2tensor(D)[:,None]
        
        self.forward_model.to(device)
        x = torch.cat((J, D), dim=1).to(device)
        y = self.forward_model(x)

        # y = torch.tensor([[ 6.1967, 15.8520,  5.3372,  3.6628]])

        omega, inten = torch.split(y, (y.shape[1]//2, y.shape[1]//2), dim=1)
        t_extended = [torch.linspace(_t-self.pulse_width/2, _t+self.pulse_width/2, int(self.pulse_width/0.01)).to(device) for _t in t]
        # S_pred = torch.zeros((x.shape[0],1,t.shape[0]))
        # for i_t, _t in enumerate(t):
        #     t_extended.append(
        #         torch.linspace(_t-self.pulse_width/2, _t+self.pulse_width/2, int(self.pulse_width/0.01)).to(device))
        t_extended = torch.vstack(t_extended).flatten().to(device)
        S_envelope = torch.exp(-torch.einsum("bm,nt->bmnt", F.relu(gamma), t_extended.abs().reshape(len(t),-1)))
        I_pred = torch.trapz(
            (jit_batch_spec_to_Sqt(
                omega, inten, t_extended, self.meV_to_2piTHz
                ).sum(dim=1, keepdim=True).reshape(omega.shape[0],1,len(t),-1) * S_envelope).abs().pow(2), 
                t_extended.reshape(len(t),-1), dim=-1) / self.pulse_width

        if ret_tensor:
            return I_pred.squeeze()
        else:
            return I_pred.detach().cpu().squeeze().numpy()

    def step_OptBayesExpt(self, func_measure):
        # next_setting = self.obe_model.opt_setting()
        next_setting = self.obe_model.get_setting()
        if self.obe_model.selection_method in ['optimal', 'good']:
            self.utility_list.append(self.obe_model.utility_stored)
        else:
            self.utility_list.append(None)
        next_observable = func_measure.simdata(next_setting)
        
        measurement = (next_setting, next_observable, func_measure.noise_level)
        if self.reference_setting_value is not None:
            ref_setting, ref_value = self.reference_setting_value
            ref_value_denum = self.obe_model.eval_over_all_parameters(ref_setting)[0].mean()
            scale_factor = ref_value / ref_value_denum
        else:
            scale_factor = None
        self.obe_model.pdf_update(measurement, scale_factor=scale_factor)

        self.measured_settings.append(next_setting)
        self.measured_observables.append(next_observable)
        self.param_mean.append(self.obe_model.mean())
        self.param_std.append(self.obe_model.std())
        # pars = [np.random.normal(self.obe_model.mean()[i], self.obe_model.std()[i], num_samples_per_step) for i in range(len(self.obe_model.mean()))]
        _, model_predictions_on_obe_mean = self.predict_all_settings()
        self.model_predictions_on_obe_mean.append(model_predictions_on_obe_mean)
            # if np.all(param_std[-1] < 1e-2):
            #     break

    def run_N_steps_OptBayesExpt(
        self, N, func_measure, 
        steps_on_maximization=None, ret_particles=False, verbose=False):
        if ret_particles:
            particles = [self.obe_model.particles.copy()]
            particle_weights = [self.obe_model.particle_weights.copy()]
        if verbose:
            print(f"using the {self.obe_model.selection_method} setting")
            pbar = tqdm(range(N), desc="Running OptBayesExpt")
        else:
            pbar = range(N)
        for i in pbar:
            self.step_OptBayesExpt(func_measure)
            if steps_on_maximization is not None:
                # _, _ = self.run_gradient_desc_on_current_measurements(
                #     steps_on_maximization, lr=0.001, init_bayes_guess=True, batch_size=self.obe_model.particles.shape[1])
                # self.update_OptBayesExpt_particles(update_weights=False)
                # print('updated')
                for j in range(steps_on_maximization):
                    self.step_Maximization()
            if ret_particles:
                particles.append(self.obe_model.particles.copy())
                particle_weights.append(self.obe_model.particle_weights.copy())
        if ret_particles:
            return np.asarray(particles), np.asarray(particle_weights)
        

    def run_N_steps_OptBayesExpt_w_GD(
        self, N, func_measure, N_GD=100, lr=1e-2, gd_seperation=20, ret_particles=False, verbose=False, reference_noise_level=10):
        if ret_particles:
            particles = [self.obe_model.particles.copy()]
            particle_weights = [self.obe_model.particle_weights.copy()]
        if verbose:
            print(f"using the {self.obe_model.selection_method} setting")
            pbar = tqdm(range(N), desc="Running OptBayesExpt")
        else:
            pbar = range(N)
        last_gd_step = 0
        for i in pbar:
            self.step_OptBayesExpt(func_measure)
            if (self.estimate_error() > 5 * reference_noise_level) and (i > last_gd_step + gd_seperation):
                print("running GD")
                self.run_gradient_desc_on_current_measurements(
                    N_GD, lr=lr, batch_size=self.obe_model.n_particles, init_bayes_guess=False)
                self.update_OptBayesExpt_particles()
                last_gd_step = i
            # else:
            #     print(self.estimate_error(), i, last_gd_step)
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

    def estimate_error(self, ):
        unique_settings, mean_observables, std_observables = \
            self.get_organized_measurements()
        if len(unique_settings) == 0:
            return None
        t = torch.from_numpy(unique_settings).squeeze()
        Y_true = torch.from_numpy(mean_observables).squeeze()
        particle_weights = torch.from_numpy(self.obe_model.particle_weights)
        particles = torch.from_numpy(self.obe_model.particles)
        I_pred = self.model_function_conv((t,), particles, (), ret_tensor=True)
        if self.reference_setting_value is not None:
            ref_setting, ref_value = self.reference_setting_value
            I0_pred = self.model_function_conv(ref_setting, particles, (), ret_tensor=True)
            scale_factor = ref_value / I0_pred.mean()
            I_pred = I_pred * scale_factor
        if I_pred.ndim == 1:
            I_pred.unsqueeze_(1)
        I_pred_mean = torch.einsum("nt, n -> t", I_pred, particle_weights)
        loss = F.mse_loss(I_pred_mean, Y_true.to(I_pred_mean))
        return loss.item()

    def step_Maximization(self, max_step=1, lr=1e-7, min_datapoints=1):
        unique_settings, mean_observables, std_observables = \
            self.get_organized_measurements()
        if len(unique_settings) >= min_datapoints:
            t = torch.from_numpy(unique_settings).squeeze()
            Y_true = torch.from_numpy(mean_observables).squeeze()
            particle_weights = torch.from_numpy(self.obe_model.particle_weights)
            particles = torch.from_numpy(self.obe_model.particles)

            for step in range(max_step):
                particles.requires_grad_(True)
                particle_weights.requires_grad_(True)
                I_pred = self.model_function_conv((t,), particles, (), ret_tensor=True)
                if self.reference_setting_value is not None:
                    ref_setting, ref_value = self.reference_setting_value
                    I0_pred = self.model_function_conv(ref_setting, particles, (), ret_tensor=True)
                    scale_factor = ref_value / I0_pred.mean()
                    I_pred = I_pred * scale_factor
                if I_pred.ndim == 1:
                    I_pred.unsqueeze_(1)
                I_pred_mean = torch.einsum("nt, n -> t", I_pred, particle_weights)
                loss = F.mse_loss(I_pred_mean, Y_true.to(I_pred_mean))
                particles_grad, particle_weights_grad = torch.autograd.grad(loss, (particles, particle_weights))
                particles = particles - 1e-3 * particles_grad
                _particle_weights = torch.relu(particle_weights - lr * particle_weights_grad)
                if np.abs(_particle_weights.sum().item()) < 1e-3:
                    break
                else:
                    particle_weights.requires_grad_(False)
                    particle_weights.data[:] = (_particle_weights.clone() / _particle_weights.sum()).data
                
            self.obe_model.particle_weights = particle_weights.detach().cpu().numpy()

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
    
    # def update_OptBayesExpt_particles(self,):
    #     particles = torch.cat([eval(f'self.forward_model.{name}.data').T for name in ('J', 'D', 'gamma')], dim=0).cpu().numpy()
    #     self.obe_model.particles = particles
    #     self.obe_model.particle_weights = np.ones(self.obe_model.n_particles) / self.obe_model.n_particles
    def update_OptBayesExpt_particles(self, update_weights=True):
        particles = torch.cat([self.forward_model.J.data.T, self.forward_model.D.data.T, self.forward_model.gamma.data.T], dim=0).cpu().numpy()
        self.obe_model.particles = particles
        if update_weights:
            self.obe_model.particle_weights = np.ones(self.obe_model.n_particles) / self.obe_model.n_particles