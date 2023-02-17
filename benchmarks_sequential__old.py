# %%
import itertools
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from src.model_specpred import SpectrumPredictor
from src.utils_model import lorentzian, spec_to_Sqt, batch_spec_to_Sqt

from matplotlib.colors import to_rgb, to_rgba

from matplotlib.colors import Normalize
from scipy.interpolate import interp1d
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import os
import pickle

# %%
# from multiprocessing import Process, Manager, Pool
from torch.multiprocessing import Pool, Manager, Process, set_start_method
# set_start_method('spawn')
from functools import partial
from itertools import repeat

from src.utils_general import prepare_sample
import optbayesexpt as obe
from src.utils_model import construct_fc_net, array2tensor, tensor2array
from src.utils_convolution import interp_nb, get_I_conv
from src.bayes import BayesianInference, jit_batch_spec_to_Sqt
import warnings
warnings.filterwarnings('ignore')

torch.set_default_dtype(torch.float64)

def measure_function(sets, pars, cons, func):
    """ Evaluates a trusted model of the experiment's output
    The equivalent of a fit function. The argument structure is
    required by OptBayesExpt.
    Args:
        sets: A tuple of setting values, or a tuple of settings arrays
        pars: A tuple of parameter arrays or a tuple of parameter values
        cons: A tuple of floats
    Returns:  the evaluated function
    """
    # unpack model parameters
    t, = sets
    if isinstance(t, (int, float)):
        t = np.array([t,])
    else:
        t = np.atleast_1d(tensor2array(t))
    I_pred = func(t)
    return I_pred

def updata_dict_for_idx(idx, X, Y, model,
                        settings, parameters, times, gamma, pulse_width, noise_level,
                        selection_method, normalize_to_value, N_steps_bayes, TASK_NAME, device):
    # print(idx)
    d = {}
    param_true, func_I_conv, func_I_noconv = prepare_sample(
            X[idx], Y[idx], gamma, times, pulse_width=pulse_width, 
            visualize=True, normalize_to_value=normalize_to_value)
    obe_sim = obe.MeasurementSimulator(
        lambda s, p, c: measure_function(s, p, c, func_I_conv), param_true, (), noise_level=noise_level)
    
    bayes = BayesianInference(
        model, settings, parameters,
        pulse_width=pulse_width, reference_setting_value=((0,), normalize_to_value),
        model_uncertainty=False, device=device)
    bayes.obe_model.set_selection_method(selection_method)

    if TASK_NAME == 'gd':
        particles_hist, p_weights_hist, errors = bayes.run_N_steps_OptBayesExpt_w_GD(
            N_steps_bayes, obe_sim, N_GD=100, lr=0.005, ret_particles=True, verbose=False, 
            gd_seperation=25, error_criterion=200)
    else:
        particles_hist, p_weights_hist, errors = bayes.run_N_steps_OptBayesExpt_wo_GD(
            N_steps_bayes, obe_sim, ret_particles=True, verbose=False)
        
    param_mean = np.asarray(bayes.param_mean)
    param_std = np.asarray(bayes.param_std)

    d['param_mean'] = param_mean
    d['param_std'] = param_std
    d['param_true'] = param_true

    d['measurement_errors'] = np.asarray(errors)
    _measurement_settings, _measurements = bayes.get_all_measurements()
    d['measurement_settings'] = _measurement_settings
    d['measurements'] = _measurements

    d['particles'] = particles_hist
    d['particle_weights'] = p_weights_hist
    return d

#%%
# if __name__ == '__main__':

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

import seaborn
palette_crest = seaborn.color_palette(palette='crest')
palette_flare = seaborn.color_palette(palette='flare')

data = torch.load("data/CrI3/20221110.pt")
X = data['param'][:,:2]
Y = torch.cat((data['omega'], data['inten']), dim=1)

indices_dict = torch.load("data_splitting/indices_42_800-100-100.pt")
test_indices = indices_dict['test']

X_test = X[test_indices]
Y_test = Y[test_indices]

print("print some values for further reference:")
print("testing:\n", X_test[:5])
model_spec = SpectrumPredictor.load_from_checkpoint("production_models/version_large_training_set/checkpoints/epoch=8456-step=422850.ckpt")
from tqdm import tqdm

RUN_NUMBERs = ['RUN_1', 'RUN_2', 'RUN_3', 'RUN_4', 'RUN_5']
TASK_NAMEs = ['gd', 'baseline', 'random', 'sequential']

gamma = 0.1
pulse_width = 0.0
noise_level = 1.0
N_steps_bayes = 100
normalize_to_value = 100
NUM_SAMPLES = len(X_test)
NUM_SAMPLES = 2
NUM_WORKERS = 5
print(f"task for pulse_width {pulse_width} and noise_level {noise_level} with {NUM_WORKERS} workers")

times = np.arange(0, 10, 0.02)
parameters = (
    np.random.uniform(-3.0, -1.0, 1001),
    np.random.uniform(-1.0,  0.0, 1001),
    np.random.uniform( 0.0,  1.0, 1001)
    )

def perform_task(TASK_NAME, RUN_NUMBER):

    if TASK_NAME in ['baseline', 'gd']:
        selection_method = 'optimal'
        settings = (times, )
    elif TASK_NAME == 'random':
        selection_method = 'random'
        settings = (times, )
    elif TASK_NAME == 'sequential':
        selection_method = 'sequential'
        settings = (np.linspace(times.min(), times.max(), N_steps_bayes), )

    SAVE_NAME = f"bayesian_{TASK_NAME}_pw-{pulse_width}_nl-{noise_level}_Nb-{N_steps_bayes}"
    print(f"SAVE_NAME is {SAVE_NAME}")

    func = partial(updata_dict_for_idx, X=X_test, Y=Y_test,
                model=model_spec, parameters=parameters, settings=settings, normalize_to_value=normalize_to_value, N_steps_bayes=N_steps_bayes,
                times=times, gamma=gamma, pulse_width=pulse_width, noise_level=noise_level, selection_method=selection_method, TASK_NAME=TASK_NAME, device=device)
    d = {}
    for i_p in tqdm(range(NUM_SAMPLES), total=NUM_SAMPLES):
        d[i_p] = func(i_p)

    SAVE_DIR = f'benchmarks/{RUN_NUMBER}'
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    with open(f'{SAVE_DIR}/{SAVE_NAME}.pkl', 'wb') as f:
        pickle.dump(d, f)

#%%

for RUN_NUMBER in RUN_NUMBERs:
    for TASK_NAME in TASK_NAMEs:
        perform_task(TASK_NAME, RUN_NUMBER)

