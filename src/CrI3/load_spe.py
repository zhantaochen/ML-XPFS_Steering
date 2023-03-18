import numpy as np
import os
def load_spe(fname):
    HH0=np.loadtxt(fname,skiprows=2,max_rows=9)
    HH0=HH0.flatten('C')
    HH0=HH0[:-1]
    E=np.loadtxt(fname,skiprows=12,max_rows=10)
    E=E.flatten('C')
    E=np.append(E,np.loadtxt(fname,skiprows=22,max_rows=1))
    E=E[:-1]
    i=24
    while i<1727:
        Ztemp=np.loadtxt(fname,skiprows=i,max_rows=10)
        Ztemp=Ztemp.flatten('C')
        Errtemp=np.loadtxt(fname,skiprows=i+12,max_rows=10)
        Errtemp=Errtemp.flatten('C')
        Ztemp=np.append(Ztemp,np.loadtxt(fname,skiprows=i+10,max_rows=1))
        Errtemp=np.append(Errtemp,np.loadtxt(fname,skiprows=i+22,max_rows=1))
        Ztemp=Ztemp.reshape((-1,1))
        Errtemp=Errtemp.reshape((-1,1))
        if i==24:
            Z=Ztemp
            Err=Errtemp
        else:
            Z=np.append(Z,Ztemp,axis=1)
            Err=np.append(Err,Errtemp,axis=1)
        i+=24
    del Ztemp,Errtemp
    return HH0, E, Z, Err