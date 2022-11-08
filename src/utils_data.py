import glob, os
import numpy as np
import scipy
import torch
from tqdm import tqdm


def mat_to_pt(mat_folder, pt_fname):
    file_lst = glob.glob(os.path.join(mat_folder, '*.mat'))
    print(f"Find {len(file_lst)} files.")
    q_idx = 0
    param_lst = []
    omega_lst = []
    inten_lst = []
    for i, file in tqdm(enumerate(file_lst), total=len(file_lst)):
        data = scipy.io.loadmat(file)
        if i == 0:
            print(f'Using the k-point ', data['hkl'][:,q_idx])
        param = torch.tensor([data["J1"][0,0], data["DM"][0,0], data["K"][0,0]])
        nz_idx = np.nonzero(data['swConv'][:,q_idx])[0]
        omega = torch.from_numpy(data['Evect'][0, nz_idx])
        inten = torch.from_numpy(data['swConv'][nz_idx, q_idx])

        if len(omega) == 1:
            omega = torch.cat((omega, omega), dim=0)
        if len(inten) == 1:
            inten = torch.cat((inten/2, inten/2), dim=0)

        param_lst.append(param)
        omega_lst.append(omega)
        inten_lst.append(inten)
    save_dict = {
        "param": torch.vstack(param_lst),
        "omega": torch.vstack(omega_lst),
        "inten": torch.vstack(inten_lst),
        "note": "J1=[-2.5,0]; DM=[-1.0,0]; K=[-6.0 0]; Jc=-0.6; Dz=-0.1; All zeros otherwise;"
    }

    torch.save(save_dict, pt_fname)