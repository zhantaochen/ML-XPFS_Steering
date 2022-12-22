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

    def generate_new_data(self, J_list, D_list, K_list, mat_folder, pt_fname):
        if not os.path.exists(mat_folder):
            os.makedirs(mat_folder)
        import matlab.engine
        eng = matlab.engine.start_matlab()
        eng.addpath(r'src/MATLAB/', nargout=0)
        print(f"\nGenerating {len(J_list)} new samples.")
        eng.generate_data(J_list,D_list,K_list, mat_folder, nargout=0)
        eng.quit()
        print(f"\nFinished sample generation.")
        mat_to_pt(mat_folder, pt_fname)

    def train_on_data(self, pt_fname, maxiter=500, global_dataset=None):

        trainer = pl.Trainer(
            max_epochs=maxiter, accelerator="gpu",
            callbacks=[TQDMProgressBar(refresh_rate=5)],
            log_every_n_steps=2, devices=1, 
            enable_checkpointing=False,
            default_root_dir="training_logs",
        )
        data = torch.load(pt_fname)
        X = data['param'][:,:2]
        Y = torch.cat((data['omega'], data['inten']), dim=1)
        if global_dataset is not None:
            X_global, Y_global = global_dataset[np.random.choice(len(global_dataset), 5 * len(X))]
            X = torch.cat((X[:,:2], X_global[:,:2]), dim=0)
            Y = torch.cat((Y, Y_global), dim=0)
        train_dataset = TensorDataset(X[:,:2], Y)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        trainer.fit(self, train_dataloader)

    def fit_measurement(
        self, t, S, 
        batch_size=10, maxiter=100, lr=0.001,
        global_dataset=None, retrain_criteria=None, supp_data_folder='./',
        J_bounds=(-2.5,0.0), D_bounds=(-1.0,0.0), K_bounds=(-1.0,0.0),
        replace_worst_with_mean=True, save_param_hist=True
    ):
        """_summary_

        Parameters
        ----------
        t : _type_
            _description_
        S : _type_
            _description_
        batch_size : int, optional
            _description_, by default 10
        maxiter : int, optional
            _description_, by default 100
        lr : float, optional
            _description_, by default 0.001
        retrain_criteria : tuple (N, M), optional
            if loss does not decrease by M in N steps, perform new training, by default None

        Returns
        -------
        _type_
            _description_
        """
        if not os.path.exists(supp_data_folder):
            os.makedirs(supp_data_folder)
        # relaxation time of magnon
        self.register_parameter("gamma", torch.nn.Parameter(torch.rand(batch_size, self.num_mode)))
        self.register_parameter("J", torch.nn.Parameter(torch.DoubleTensor(batch_size, 1).uniform_(J_bounds[0], J_bounds[1])))
        self.register_parameter("D", torch.nn.Parameter(torch.DoubleTensor(batch_size, 1).uniform_(D_bounds[0], D_bounds[1])))
        self.register_parameter("K", torch.nn.Parameter(torch.DoubleTensor(batch_size, 1).uniform_(K_bounds[0], K_bounds[1])))
        
        if save_param_hist:
            self.gamma_hist = [self.gamma.data]
            self.J_hist = [self.J.data]
            self.D_hist = [self.D.data]
            self.K_hist = [self.K.data]

        optimizer = torch.optim.Adam([self.gamma, self.J, self.D, self.K], lr=lr)

        if retrain_criteria is not None:
            retrain_diff, retrain_step, N_new_samples = retrain_criteria
            retrain_counter = 0
            last_retrain_iter = 0
        loss_hist = []
        pbar = tqdm(range(maxiter))
        for i_iter in pbar:
            # x = torch.cat((self.J, self.D, self.K), dim=1)
            x = torch.cat((self.J, self.D), dim=1)
            y = self.fc_net(x.to(self.device))
            omega, inten = torch.split(y, [self.num_mode, self.num_mode], dim=1)
            # batch x mode x time
            S_envelope = torch.exp(-torch.einsum("bm,t->bmt", F.relu(self.gamma), t))
            S_pred = (batch_spec_to_Sqt(omega, inten, t) * S_envelope).sum(dim=1)
            S_pred = S_pred / S_pred[:,0,None] * S[0]
            loss_batch = (S_pred - torch.atleast_2d(S).repeat_interleave(batch_size,dim=0).to(S_pred)).pow(2).mean(dim=1)
            loss = loss_batch.mean()
            # loss = F.mse_loss(S_pred, torch.atleast_2d(S).repeat_interleave(batch_size,dim=0).to(S_pred))
            
            if loss < 0.001 * S[0]:
                break
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_hist.append(loss.item())
            pbar.set_description(f"Iter {i_iter:4d} Loss {loss.item():4f}")
            
            
            if replace_worst_with_mean and ((loss_batch.max().abs() - loss_batch.min().abs())/loss_batch.min().abs() > 5.0):
                idx_loss_descending = torch.argsort(loss_batch, descending=True)
                idx_worst = idx_loss_descending[:2]
                idx_best =  idx_loss_descending[-2:]
                # mask = torch.ones(batch_size, dtype=torch.bool)
                # mask[idx_worst] = False
                with torch.no_grad():
                    self.gamma.data[idx_worst] = self.gamma.data[idx_best].mean(dim=0)
                    self.J.data[idx_worst] = self.J.data[idx_best].mean() + torch.randn_like(self.J.data[idx_worst]) * 0.01
                    self.D.data[idx_worst] = self.D.data[idx_best].mean() + torch.randn_like(self.J.data[idx_worst]) * 0.01
                    self.K.data[idx_worst] = self.K.data[idx_best].mean() + torch.randn_like(self.J.data[idx_worst]) * 0.01

            # with torch.no_grad():
            #     self.J.data[:] = self.J.data.clamp_(J_bounds[0], J_bounds[1])
            #     self.D.data[:] = self.D.data.clamp_(D_bounds[0], D_bounds[1])
            #     self.K.data[:] = self.K.data.clamp_(K_bounds[0], K_bounds[1])

                
            if save_param_hist:
                self.gamma_hist.append(self.gamma.data.clone())
                self.J_hist.append(self.J.data.clone())
                self.D_hist.append(self.D.data.clone())
                self.K_hist.append(self.K.data.clone())

            # retrain model on nearby points if loss stagnates
            if (retrain_criteria is not None) and \
                (i_iter > retrain_step+last_retrain_iter) and \
                (loss_hist[i_iter] - loss_hist[i_iter-retrain_step+2] > -retrain_diff) and \
                (loss.item() > retrain_diff):
                # with torch.no_grad():
                #     loss_batch_wise = (S_pred, torch.atleast_2d(S).repeat_interleave(batch_size,dim=0).to(S_pred)).pow(2).mean(dim=-1)
                print("need new data")
                print(self.J.mean(), self.D.mean())
                J_lst = (
                    self.J.data.repeat_interleave(N_new_samples) + 
                    torch.empty_like(self.J.data.repeat_interleave(N_new_samples)).normal_(0.0, 0.5).to(self.J)
                ).detach().cpu().clamp_max_(0.0).numpy()
                D_lst = (
                    self.D.data.repeat_interleave(N_new_samples) + 
                    torch.empty_like(self.D.data.repeat_interleave(N_new_samples)).normal_(0.0, 0.5).to(self.D)
                ).detach().cpu().clamp_max_(0.0).numpy()
                K_lst = (
                    self.K.data.repeat_interleave(N_new_samples) + 
                    torch.empty_like(self.K.data.repeat_interleave(N_new_samples)).normal_(0.0, 0.5).to(self.K)
                ).detach().cpu().clamp_max_(0.0).numpy()

                self.generate_new_data(
                    J_lst, D_lst, K_lst,
                    mat_folder=os.path.join(supp_data_folder, f"supp_data_{retrain_counter}/"),
                    pt_fname=os.path.join(supp_data_folder, f"supp_data_{retrain_counter}/", f"supp_data_{retrain_counter}.pt")
                )
                self.train_on_data(
                    os.path.join(supp_data_folder, 
                    f"supp_data_{retrain_counter}/", f"supp_data_{retrain_counter}.pt"),
                    global_dataset=global_dataset    
                )
                retrain_counter += 1
                last_retrain_iter = i_iter

        return loss_hist

    def fit_measurement_with_OptBayesExpt_parameters(
        self, t, S, params,
        batch_size=10, maxiter=100, lr=0.001,
        replace_worst_with_mean=True, save_param_hist=True
    ):
        """_summary_

        Parameters
        ----------
        t : _type_
            _description_
        S : _type_
            _description_
        batch_size : int, optional
            _description_, by default 10
        maxiter : int, optional
            _description_, by default 100
        lr : float, optional
            _description_, by default 0.001
        retrain_criteria : tuple (N, M), optional
            if loss does not decrease by M in N steps, perform new training, by default None

        Returns
        -------
        _type_
            _description_
        """
        t = array2tensor(t)
        S = array2tensor(S)
        
        for (name, mean, std) in zip(*params):
            param = mean + std*torch.randn(batch_size,1)
            self.register_parameter(name, torch.nn.Parameter(param))
        param_lst = []
        for name in params[0]:
            param_lst.append({'params': eval(f'self.{name}')})
        optimizer = torch.optim.Adam(param_lst, lr=lr)

        loss_hist = []
        if save_param_hist: 
            self.param_hist = {name: [] for name in params[0]}
        pbar = tqdm(range(maxiter))
        for i_iter in pbar:
            # x = torch.cat((self.J, self.D, self.K), dim=1)
            x = torch.cat((self.J, self.D), dim=1)
            y = self.fc_net(x.to(self.device))
            omega, inten = torch.split(y, [self.num_mode, self.num_mode], dim=1)
            # batch x mode x time
            S_envelope = torch.exp(-torch.einsum("bm,t->bmt", F.relu(self.gamma), t))
            S_pred = (batch_spec_to_Sqt(omega, inten, t) * S_envelope).sum(dim=1)
            S_pred = S_pred / S_pred[:,0,None] * S[0]
            loss_batch = (S_pred - torch.atleast_2d(S).repeat_interleave(batch_size,dim=0).to(S_pred)).pow(2).mean(dim=1)
            loss = loss_batch.mean()
            # loss = F.mse_loss(S_pred, torch.atleast_2d(S).repeat_interleave(batch_size,dim=0).to(S_pred))
            
            if loss < 0.001 * S[0]:
                break
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_hist.append(loss.item())
            pbar.set_description(f"Iter {i_iter:4d} Loss {loss.item():4f}")
            
            # if replace_worst_with_mean and ((loss_batch.max().abs() - loss_batch.min().abs())/loss_batch.min().abs() > 5.0):
            #     idx_loss_descending = torch.argsort(loss_batch, descending=True)
            #     idx_worst = idx_loss_descending[:2]
            #     idx_best =  idx_loss_descending[-2:]
                # with torch.no_grad():
                #     self.gamma.data[idx_worst] = self.gamma.data[idx_best].mean(dim=0)
                #     self.J.data[idx_worst] = self.J.data[idx_best].mean() + torch.randn_like(self.J.data[idx_worst]) * 0.01
                #     self.D.data[idx_worst] = self.D.data[idx_best].mean() + torch.randn_like(self.J.data[idx_worst]) * 0.01
                #     self.K.data[idx_worst] = self.K.data[idx_best].mean() + torch.randn_like(self.J.data[idx_worst]) * 0.01
            # print(self.J)
            if save_param_hist: 
                for name in params[0]:
                    self.param_hist[name].append(eval(f'self.{name}.clone().detach().cpu()'))
        for key in self.param_hist.keys():
            self.param_hist[key] = torch.cat(self.param_hist[key], dim=-1).T
        return loss_hist, self.param_hist

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.fc_net(x)

        loss = F.mse_loss(y_pred, y)
        
        self.log('train_loss', loss.item())

        return loss

