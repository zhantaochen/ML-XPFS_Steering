import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import scipy.constants as const

from .utils_data import mat_to_pt
from .utils_model import construct_fc_net, batch_spec_to_Sqt
from tqdm import tqdm

class SpectrumPredictor(pl.LightningModule):
    def __init__(self, num_mode=2):
        super().__init__()
        self.save_hyperparameters()
        # estimated number of magnon to be predicted
        self.num_mode = num_mode
        # spectrum predicting network
        self.fc_net = construct_fc_net(
            feat_in=3, feat_out=2*self.num_mode, feat_hid_list=None
        )
        # unit conversion, this way time is in [ps]
        self.meV_to_2piTHz = 2 * np.pi * 1e-15 / const.physical_constants['hertz-electron volt relationship'][0]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x):
        return self.fc_net(x)

    def generate_new_data(self, J_list, D_list, K_list, mat_folder, pt_fname):
        import matlab.engine
        eng = matlab.engine.start_matlab()
        eng.addpath(r'src/MATLAB/', nargout=0)
        eng.generate_data(J_list,D_list,K_list, mat_folder, nargout=0)
        eng.quit()
        mat_to_pt(mat_folder, pt_fname)

    def train_on_data(self, pt_fname, maxiter=1000):

        trainer = pl.Trainer(
            max_epochs=maxiter, accelerator="gpu",
            callbacks=[TQDMProgressBar(refresh_rate=5)],
            log_every_n_steps=2, devices=1, 
            enable_checkpointing=True,
            default_root_dir="training_logs"
        )
        data = torch.load(pt_fname)
        X = data['param']
        Y = torch.cat((data['omega'], data['inten']), dim=1)
        train_dataset = TensorDataset(X, Y)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        trainer.fit(self, train_dataloader)


    def fit_measurement(self, t, S, batch_size=10, maxiter=100, lr=0.001):
        # relaxation time of magnon
        self.register_parameter("gamma", torch.nn.Parameter(torch.rand(batch_size, self.num_mode)))
        self.register_parameter("J", torch.nn.Parameter(torch.DoubleTensor(batch_size, 1).uniform_(-2.5, 0)))
        self.register_parameter("D", torch.nn.Parameter(torch.DoubleTensor(batch_size, 1).uniform_(-1.0, 0)))
        self.register_parameter("K", torch.nn.Parameter(torch.DoubleTensor(batch_size, 1).uniform_(-6.0, 0)))
        
        optimizer = torch.optim.Adam([self.gamma, self.J, self.D, self.K], lr=lr)

        loss_hist = []
        pbar = tqdm(range(maxiter))
        for i_iter in pbar:
            x = torch.cat((self.J, self.D, self.K), dim=1)
            y = self.fc_net(x)
            omega, inten = torch.split(y, [self.num_mode, self.num_mode], dim=1)
            # batch x mode x time
            S_envelope = torch.exp(-torch.einsum("bm,t->bmt", F.relu(self.gamma), t))
            S_pred = (batch_spec_to_Sqt(omega, inten, t) * S_envelope).sum(dim=1)
            S_pred = S_pred / S_pred[:,0,None] * S[0]
            loss = F.mse_loss(S_pred, torch.atleast_2d(S).repeat_interleave(batch_size,dim=0).to(S_pred))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_hist.append(loss.item())
            pbar.set_description(f"Iter {i_iter:4d} Loss {loss.item():4f}")
        return loss_hist

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.fc_net(x)

        loss = F.mse_loss(y_pred, y)
        
        self.log('train_loss', loss.item())

        return loss

class TimeSeriesPredictor(pl.LightningModule):
    def __init__(self, num_mode=2):
        super().__init__()
        self.num_mode = 2
        self.register_parameter("tau", torch.nn.Parameter(torch.rand(2)))
