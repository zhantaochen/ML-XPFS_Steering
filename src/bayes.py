from functools import partial
import optbayesexpt as obe
import numpy as np
from numpy.linalg import LinAlgError
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .utils_model import construct_fc_net, array2tensor, tensor2array, batch_spec_to_Sqt, jit_batch_spec_to_Sqt
from .utils_gd import fit_measurement_with_OptBayesExpt_parameters
from .utils_general import get_I

import scipy.constants as const
meV_to_2piTHz = torch.tensor(2 * np.pi * 1e-15 / const.physical_constants['hertz-electron volt relationship'][0])


class OptBayesExpt_CustomCost(obe.OptBayesExpt):
    def __init__(self, 
                 cost_repulse_height=1.0, cost_repulse_width=0.25, 
                 parameter_mins=(0,0,0), parameter_maxs=(1,1,1),
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_repulse_height = cost_repulse_height
        self.cost_repulse_width = cost_repulse_width
        self.parameter_mins = parameter_mins
        self.parameter_maxs = parameter_maxs
        self.reset_proposed_setting()

    def cost_estimate(self):
        bins = self.proposed_settings["setting_bin"]
        cost = np.ones_like(self.setting_indices).astype('float')
        for idx in np.nonzero(bins)[0]:
            cost += self.cost_repulse_height * bins[idx] * \
                np.squeeze(np.exp(-((self.allsettings - self.allsettings[:,idx]) / self.cost_repulse_width)**2))
        return cost
        # return 1.0

    def enforce_parameter_constraints(self):
        """A stub for enforcing constraints on parameters
        for example::
            # find the particles with disallowed parameter values
            # (negative parameter values in this example)
            bad_ones = np.argwhere(self.parameters[3] < 0)
                for index in bad_ones:
                    # setting a weight = 0 effectively eliminates the particle
                    self.particle_weights[index] = 0
            # renormalize
            self.particle_weights = self.particle_weights / np.sum(self.particle_weights)
        """
        bad_idx = []
        for i in range(len(self.parameters)):
            bad_idx += np.argwhere(self.parameters[i] < self.parameter_mins[i]).tolist() + np.argwhere(self.parameters[i] > self.parameter_maxs[i]).tolist()
        bad_idx = np.unique(bad_idx)
        if len(bad_idx) > 0:
            self.parameters[:,bad_idx] = np.vstack([
                self.parameter_mins[i] + (self.parameter_maxs[i]-self.parameter_mins[i]) * np.random.rand(len(bad_idx)) for i in range(len(self.parameters))])

class BayesianInference:
    NUM_SAMPLES_PER_STEP = 10
    meV_to_2piTHz = torch.tensor(2 * np.pi * 1e-15 / const.physical_constants['hertz-electron volt relationship'][0])
    def __init__(
        self, forward_model, settings=(), parameters=(), constants=(), 
        pulse_width=None, reference_setting_value=None, model_uncertainty=False, 
        parameter_mins=(0,0,0), parameter_maxs=(1,1,1),
        cost_repulse_height=1.0, cost_repulse_width=0.25, device='cpu'
    ):
        # super().__init__()
        # self.register_module('forward_model', model)
        self.forward_model = forward_model
        self.settings = settings
        self.parameters = parameters
        self.constants = constants
        self.pulse_width = pulse_width
        self.model_uncertainty = model_uncertainty
        self.cost_repulse_height = cost_repulse_height
        self.cost_repulse_width = cost_repulse_width
        self.parameter_mins = parameter_mins
        self.parameter_maxs = parameter_maxs

        if device is None:
            self.device = 'cuda' if torch.cuda.is_exist() else 'cpu'
        else:
            self.device = device

        self.reference_setting_value = reference_setting_value
        if self.pulse_width is not None:
            self.model_function = partial(self.model_function_conv, ret_tensor=False, norm_I0=100, device=device)
        else:
            self.model_function = partial(self.model_function_noconv, ret_tensor=False, norm_I0=100, device=device)
        self.init_OptBayesExpt()
        self.init_saving_lists()



    def init_OptBayesExpt(self, ):
        self.obe_model = OptBayesExpt_CustomCost(
            measurement_model=self.model_function, setting_values=self.settings, parameter_samples=self.parameters, 
            constants=self.constants, utility_method='variance_full',
            cost_repulse_height=self.cost_repulse_height, cost_repulse_width=self.cost_repulse_width,
            parameter_mins=self.parameter_mins, parameter_maxs=self.parameter_maxs)
        # self.obe_model = OptBayesExpt_CustomCost(
        #     measurement_model=self.model_function, setting_values=self.settings, parameter_samples=self.parameters, 
        #     constants=self.constants, utility_method='variance_approx',
        #     cost_repulse_height=0.25, cost_repulse_width=0.05)
        
        # self.obe_model = obe.OptBayesExpt(
        #     measurement_model=self.model_function, setting_values=self.settings, parameter_samples=self.parameters, 
        #     constants=self.constants, utility_method='variance_full')
        # self.obe_model = obe.OptBayesExpt(
        #     measurement_model=self.model_function, setting_values=self.settings, parameter_samples=self.parameters, 
        #     constants=self.constants, utility_method='variance_approx')
        
        # self.obe_model = obe.OptBayesExpt(measurement_model=self.model_function, setting_values=self.settings, parameter_samples=self.parameters, constants=self.constants)
        
    def init_saving_lists(self, ):
        self.measured_settings = []
        self.measured_observables = []
        self.param_mean = [self.obe_model.mean()]
        self.param_std = [self.obe_model.std()]
        self.utility_list = []
        self.model_predictions_on_obe_mean = []

    def handle_inputs(self, sets, pars, cons=None, device='cpu'):
        t, = sets
        J, D, gamma, elas_amp, elas_wid = pars
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

        if isinstance(elas_amp, (int, float)):
            elas_amp = torch.atleast_2d(torch.tensor([elas_amp]))
        else:
            elas_amp = array2tensor(elas_amp)[:,None]
        elas_amp = elas_amp.to(device)
        
        if isinstance(elas_wid, (int, float)):
            elas_wid = torch.atleast_2d(torch.tensor([elas_wid]))
        else:
            elas_wid = array2tensor(elas_wid)[:,None]
        elas_wid = elas_wid.to(device)

        if isinstance(J, (int, float)):
            J = torch.tensor([[J]])
            D = torch.tensor([[D]])
        else:
            J = array2tensor(J)[:,None]
            D = array2tensor(D)[:,None]
        return t, J, D, gamma, elas_amp, elas_wid

    def model_function_noconv(self, sets, pars, cons, ret_tensor=False, norm_I0=100):
        """ Evaluates a trusted model of the experiment's output
        The equivalent of a fit function. The argument structure is
        required by OptBayesExpt.
        Args:
            sets: A tuple of setting values, or a tuple of settings arrays
            pars: A tuple of parameter arrays or a tuple of parameter values
            cons: A tuple of floats
        Returns:  the evaluated function
        """
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'cpu'
        t, J, D, gamma = self.handle_inputs(sets, pars, device=device)
        self.forward_model.to(device)
        x = torch.cat((J, D), dim=1).to(device)
        if not self.model_uncertainty:
            y = self.forward_model(x)
        else:
            y = self.forward_model(x)
        omega, inten = torch.split(y, (y.shape[1]//2, y.shape[1]//2), dim=1)
        S_envelope = torch.exp(-torch.einsum("bm,t->bmt", F.relu(gamma), t))
        S_pred = (jit_batch_spec_to_Sqt(omega, inten, t, self.meV_to_2piTHz) * S_envelope).sum(dim=1)
        I_pred = torch.abs(S_pred)**2
        if ret_tensor:
            return I_pred.squeeze()
        else:
            return I_pred.detach().cpu().squeeze().numpy()

    def model_function_conv(self, sets, pars, cons, ret_tensor=False, norm_I0=100, device='cpu'):
        """ Evaluates a trusted model of the experiment's output
        The equivalent of a fit function. The argument structure is
        required by OptBayesExpt.
        Args:
            sets: A tuple of setting values, or a tuple of settings arrays
            pars: A tuple of parameter arrays or a tuple of parameter values
            cons: A tuple of floats
        Returns:  the evaluated function
        """
        # print(par_weights)
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # device = 'cpu'
        t, J, D, gamma, elas_amp, elas_wid = self.handle_inputs(sets, pars, device=device)
        
        if not self.model_uncertainty:
            self.forward_model.to(device)
        else:
            self.forward_model.model.to(device)
        x = torch.cat((J, D), dim=1).to(device)
        if not self.model_uncertainty:
            y = self.forward_model(x)
        else:
            y_mu, y_var = self.forward_model(x)
            y = torch.distributions.MultivariateNormal(y_mu, y_var).sample()
        # omega, inten = torch.split(y, (y.shape[1]//2, y.shape[1]//2), dim=1)
        # t_extended = [torch.linspace(_t-self.pulse_width/2, _t+self.pulse_width/2, int(self.pulse_width/0.01)).to(device) for _t in t]
        # t_extended = torch.vstack(t_extended).flatten().to(device)
        # S_envelope = torch.exp(-torch.einsum("bm,nt->bmnt", F.relu(gamma), t_extended.abs().reshape(len(t),-1)))
        # I_pred = torch.trapz(
        #     (jit_batch_spec_to_Sqt(
        #         omega, inten, t_extended, self.meV_to_2piTHz
        #         ).sum(dim=1, keepdim=True).reshape(omega.shape[0],1,len(t),-1) * S_envelope).abs().pow(2), 
        #         t_extended.reshape(len(t),-1), dim=-1) / self.pulse_width

        I_pred = get_I(t, y, gamma, 
                       torch.tensor(self.pulse_width).to(y).clone().detach(), 
                       torch.tensor(self.meV_to_2piTHz).to(y).clone().detach(),
                       elas_amp=elas_amp, elas_wid=elas_wid)
        I_pred_t0 = get_I(torch.tensor([0,]), y, gamma, 
                          torch.tensor(self.pulse_width).to(y).clone().detach(), 
                          torch.tensor(self.meV_to_2piTHz).to(y).clone().detach(),
                          elas_amp=elas_amp, elas_wid=elas_wid)
        I_out = I_pred / I_pred_t0 * norm_I0
        if ret_tensor:
            return I_out.squeeze()
        else:
            return I_out.detach().cpu().squeeze().numpy()

    def step_OptBayesExpt(self, func_measure):
        next_setting = self.obe_model.get_setting()
        if self.obe_model.selection_method in ['optimal', 'good']:
            self.utility_list.append(self.obe_model.utility_stored)
        else:
            self.utility_list.append(None)
        next_observable = func_measure.simdata(next_setting)
        
        measurement = (next_setting, next_observable, func_measure.noise_level)
        # if self.reference_setting_value is not None:
        #     ref_setting, ref_value = self.reference_setting_value
        #     ref_value_denum = self.obe_model.eval_over_all_parameters(ref_setting)[0].mean()
        #     scale_factor = ref_value / ref_value_denum
        # else:
        #     scale_factor = None
        # self.obe_model.pdf_update(measurement, scale_factor=scale_factor)
        self.obe_model.pdf_update(measurement, scale_factor=None)

        self.measured_settings.append(next_setting)
        self.measured_observables.append(next_observable)
        self.param_mean.append(self.obe_model.mean())
        self.param_std.append(self.obe_model.std())
        # pars = [np.random.normal(self.obe_model.mean()[i], self.obe_model.std()[i], num_samples_per_step) for i in range(len(self.obe_model.mean()))]
        _, model_predictions_on_obe_mean = self.predict_all_settings()
        self.model_predictions_on_obe_mean.append(model_predictions_on_obe_mean)
            # if np.all(param_std[-1] < 1e-2):
            #     break

    # def run_N_steps_OptBayesExpt(
    #     self, N, func_measure, 
    #     steps_on_maximization=None, ret_particles=False, verbose=False):
    #     if ret_particles:
    #         particles = [self.obe_model.particles.copy()]
    #         particle_weights = [self.obe_model.particle_weights.copy()]
    #     if verbose:
    #         print(f"using the {self.obe_model.selection_method} setting")
    #         pbar = tqdm(range(N), desc="Running OptBayesExpt")
    #     else:
    #         pbar = range(N)
    #     for i in pbar:
    #         self.step_OptBayesExpt(func_measure)
    #         if steps_on_maximization is not None:
    #             # _, _ = self.run_gradient_desc_on_current_measurements(
    #             #     steps_on_maximization, lr=0.001, init_bayes_guess=True, batch_size=self.obe_model.particles.shape[1])
    #             # self.update_OptBayesExpt_particles(update_weights=False)
    #             # print('updated')
    #             for j in range(steps_on_maximization):
    #                 self.step_Maximization()
    #         if ret_particles:
    #             particles.append(self.obe_model.particles.copy())
    #             particle_weights.append(self.obe_model.particle_weights.copy())
    #     if ret_particles:
    #         return np.asarray(particles), np.asarray(particle_weights)
    
    def run_N_steps_OptBayesExpt_wo_GD(
        self, N, func_measure, ret_particles=False, verbose=False):
        if ret_particles:
            particles = [self.obe_model.particles.copy()]
            particle_weights = [self.obe_model.particle_weights.copy()]
            errors = []
        if verbose:
            print(f"using the {self.obe_model.selection_method} setting")
            pbar = tqdm(range(N), desc="Running OptBayesExpt")
        else:
            pbar = range(N)
        for i in pbar:
            try:
                self.step_OptBayesExpt(func_measure)
            except LinAlgError:
                pass
            current_error = self.estimate_error()
            if ret_particles:
                particles.append(self.obe_model.particles.copy())
                particle_weights.append(self.obe_model.particle_weights.copy())
                errors.append(current_error)
        if ret_particles:
            return np.asarray(particles), np.asarray(particle_weights), errors

    def run_N_steps_OptBayesExpt_w_GD(
        self, N, func_measure, run_GD=True, N_GD=100, lr=1e-2, gd_seperation=20, error_criterion=50, 
        init_bayes_guess=False, std_criterion=1e-3, ret_particles=False, verbose=False):
        if ret_particles:
            particles = [self.obe_model.particles.copy()]
            particle_weights = [self.obe_model.particle_weights.copy()]
            errors = []
        if verbose:
            print(f"using the {self.obe_model.selection_method} setting")
            pbar = tqdm(range(N), desc="Running OptBayesExpt")
        else:
            pbar = range(N)
        last_gd_step = 0
        for i in pbar:
            try:
                self.step_OptBayesExpt(func_measure)
            except LinAlgError:
                pass

            current_error = self.estimate_error()
            if current_error is not None:
                if current_error > error_criterion:
                    run_GD = True
                else:
                    run_GD = False
            else:
                run_GD = True

            if (i >= last_gd_step + gd_seperation) and run_GD:
                if verbose:
                    print(f"running GD at step {i}, current error {current_error}")
                # print(self.obe_model.mean(), self.obe_model.std())
                loss_hist, param_hist = self.run_gradient_desc_on_current_measurements(
                    N_GD, lr=lr, batch_size=self.obe_model.n_particles, init_bayes_guess=init_bayes_guess)
                particles_to_update = tensor2array(torch.cat([param_hist[key][-1].unsqueeze(0) for key in param_hist.keys()], dim=0))
                self.update_OptBayesExpt_particles(particles_to_update)
                # print(self.obe_model.mean(), self.obe_model.std())
                last_gd_step = i
            # else:
            #     print(f"skipping GD at step {i}, current error {current_error}")
            # else:
            #     self.step_Maximization(max_step=1, lr=lr, min_datapoints=1)
                
            if ret_particles:
                particles.append(self.obe_model.particles.copy())
                particle_weights.append(self.obe_model.particle_weights.copy())
                errors.append(current_error)
        if ret_particles:
            return np.asarray(particles), np.asarray(particle_weights), errors

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
            # loss_hist, params_hist = self.forward_model.fit_measurement_with_OptBayesExpt_parameters(
            #     t, S, (('J', 'D', 'gamma'), self.obe_model.mean(), self.obe_model.std()), lr=lr,
            #     maxiter=N, batch_size=batch_size
            # )
            loss_hist, params_hist = fit_measurement_with_OptBayesExpt_parameters(
                self.forward_model, t, S, (('J', 'D', 'gamma', 'elas_amp', 'elas_wid'), torch.from_numpy(self.obe_model.particles), (None,)*5), 
                pulse_width=self.pulse_width, norm_I0=100, params_type = 'particles',
                lr=lr, maxiter=N, batch_size=self.obe_model.n_particles, model_uncertainty=self.model_uncertainty, verbose=False, device=self.device
            )
        else:
            # loss_hist, params_hist = self.forward_model.fit_measurement_with_OptBayesExpt_parameters(
            #     t, S, (('J', 'D', 'gamma'), (-2.0, -0.5, 0.5), (0.5, 0.25, 0.25)), lr=lr,
            #     maxiter=N, batch_size=batch_size
            # )
            # loss_hist, params_hist = fit_measurement_with_OptBayesExpt_parameters(
            #     self.forward_model, t, S, (('J', 'D', 'gamma'), (-2.0, -0.5, 0.5), (1/3, 1/6, 1/6)), 
            #     pulse_width=self.pulse_width, norm_I0=100, 
            #     lr=lr, maxiter=N, batch_size=self.obe_model.n_particles, model_uncertainty=self.model_uncertainty, verbose=False, device=self.device
            # )
            loss_hist, params_hist = fit_measurement_with_OptBayesExpt_parameters(
                self.forward_model, t, S, (('J', 'D', 'gamma', 'elas_amp', 'elas_wid'), torch.from_numpy(np.vstack(self.parameters)), (None,)*5), 
                pulse_width=self.pulse_width, norm_I0=100, params_type='particles',
                lr=lr, maxiter=N, batch_size=self.obe_model.n_particles, model_uncertainty=self.model_uncertainty, verbose=False, device=self.device
            )
        return loss_hist, params_hist

    def estimate_particle_error(self, particles=None):
        unique_settings, mean_observables, std_observables = \
            self.get_organized_measurements()
        if len(unique_settings) == 0:
            return None
        t = torch.from_numpy(unique_settings).squeeze()
        Y_true = torch.from_numpy(mean_observables).squeeze()
        if particles is None:
            particles = torch.from_numpy(self.obe_model.particles)
        I_pred = self.model_function_conv((t,), particles, (), ret_tensor=True)
        # if self.reference_setting_value is not None:
        #     ref_setting, ref_value = self.reference_setting_value
        #     I0_pred = self.model_function_conv(ref_setting, particles, (), ret_tensor=True)
        #     scale_factor = ref_value / I0_pred.mean()
        #     I_pred = I_pred * scale_factor
        if I_pred.ndim == 1:
            I_pred.unsqueeze_(1)
        particle_error = (I_pred - torch.atleast_1d(Y_true).unsqueeze(0).to(I_pred)).pow(2).mean(dim=-1).detach()
        return particle_error

    def estimate_error(self, particles=None, particle_weights=None):
        # unique_settings, mean_observables, std_observables = \
        #     self.get_organized_measurements()
        # if len(unique_settings) == 0:
        #     return None
        # t = torch.from_numpy(unique_settings).squeeze()
        # Y_true = torch.from_numpy(mean_observables).squeeze()
        # particle_weights = torch.from_numpy(self.obe_model.particle_weights)
        # particles = torch.from_numpy(self.obe_model.particles)
        # I_pred = self.model_function_conv((t,), particles, (), ret_tensor=True)
        # # if self.reference_setting_value is not None:
        # #     ref_setting, ref_value = self.reference_setting_value
        # #     I0_pred = self.model_function_conv(ref_setting, particles, (), ret_tensor=True)
        # #     scale_factor = ref_value / I0_pred.mean()
        # #     I_pred = I_pred * scale_factor
        # if I_pred.ndim == 1:
        #     I_pred.unsqueeze_(1)
        # I_pred_mean = torch.einsum("nt, n -> t", I_pred, particle_weights.to(I_pred))
        # loss = F.mse_loss(I_pred_mean, torch.atleast_1d(Y_true).to(I_pred_mean))
        particle_error = self.estimate_particle_error(particles)
        if particle_weights is None:
            particle_weights = torch.from_numpy(self.obe_model.particle_weights).to(particle_error)
        else:
            particle_weights = particle_weights.to(particle_error)
        if particle_error is not None:
            error = (particle_error * particle_weights).sum()
            return error.item()
        else:
            return None

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
                # if self.reference_setting_value is not None:
                #     ref_setting, ref_value = self.reference_setting_value
                #     I0_pred = self.model_function_conv(ref_setting, particles, (), ret_tensor=True)
                #     scale_factor = ref_value / I0_pred.mean()
                #     I_pred = I_pred * scale_factor
                if I_pred.ndim == 1:
                    I_pred.unsqueeze_(1)
                I_pred_mean = torch.einsum("nt, n -> t", I_pred, particle_weights.to(I_pred))
                loss = F.mse_loss(I_pred_mean, torch.atleast_1d(Y_true.to(I_pred_mean)))
                particles_grad, particle_weights_grad = torch.autograd.grad(loss, (particles, particle_weights))
                particles = particles - lr * particles_grad
                # _particle_weights = torch.relu(particle_weights - lr * particle_weights_grad)
                # if np.abs(_particle_weights.sum().item()) < 1e-3:
                #     particle_weights.requires_grad_(False)
                #     particle_weights = torch.ones_like(particle_weights) / particle_weights.shape[0]
                #     particle_weights.data[:] = (_particle_weights.clone() / _particle_weights.sum()).data
                # else:
                #     particle_weights.requires_grad_(False)
                #     particle_weights.data[:] = (_particle_weights.clone() / _particle_weights.sum()).data
            self.obe_model.particles = particles.detach().cpu().numpy()
            # self.obe_model.particle_weights = particle_weights.detach().cpu().numpy()

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
    def update_OptBayesExpt_particles(self, particles, update_weights=True):
        
        particle_error_old = self.estimate_particle_error(self.obe_model.particles)
        particle_error_new = self.estimate_particle_error(particles)
        if self.obe_model.n_particles % 2 == 1:
            separator = (self.obe_model.n_particles + 1) // 4
        self.obe_model.particles[:,particle_error_old.argsort()[-separator:]] = \
            particles[:,particle_error_new.argsort()[:separator]]
        if update_weights:
            self.obe_model.particle_weights = np.ones(self.obe_model.n_particles) / self.obe_model.n_particles