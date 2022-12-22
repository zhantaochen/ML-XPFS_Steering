import os
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import scipy.constants as const

from .utils_data import mat_to_pt
from .utils_model import construct_fc_net, batch_spec_to_Sqt, array2tensor, tensor2array
from tqdm import tqdm

class SpectrumPredictor(pl.LightningModule):
    def __init__(self, num_param_in=3, num_mode=2):
        super().__init__()
        self.save_hyperparameters()
        # estimated number of magnon to be predicted
        self.num_mode = num_mode
        # spectrum predicting network
        self.fc_net = construct_fc_net(
            feat_in=num_param_in, feat_out=2*self.num_mode, feat_hid_list=None
        )
        # unit conversion, this way time is in [ps]
        self.meV_to_2piTHz = 2 * np.pi * 1e-15 / const.physical_constants['hertz-electron volt relationship'][0]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def forward(self, x):
        return self.fc_net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.fc_net(x)

        loss = F.mse_loss(y_pred, y)
        
        self.log('train_loss', loss.item())

        return loss

