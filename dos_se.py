# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:07:01 2022

@author: James Nohl
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def load_dos_npy(compounds:list or str,xlim:float=6, plot:bool=False, path_to_dos_lib=r"G:\My Drive\Data\Simulation\Materials project\Data\220319"):
    if type(compounds) is str:
        compounds = [compounds]
    dos={}
    for comp in compounds:
        dos[comp]={}
        files = glob.glob(rf"{path_to_dos_lib}\{comp}\*.npy")
        for file in files:
            mpid = os.path.split(file)[1].split('.npy')[0]
            data = np.load(file)
            dos[comp][mpid]={}
            dos[comp][mpid]['profile'] = np.asarray([data[:,0]*-1,data[:,1]]).T
            dos[comp][mpid]['rows'] = np.where((dos[comp][mpid]['profile'][:,0]>=0)&(dos[comp][mpid]['profile'][:,0]<=xlim))[0]
            if plot is True:
                plt.plot(dos[comp][mpid]['profile'][:,0][dos[comp][mpid]['rows']],dos[comp][mpid]['profile'][:,1][dos[comp][mpid]['rows']],label=mpid)
                plt.scatter(dos[comp][mpid]['profile'][:,0][dos[comp][mpid]['rows']],dos[comp][mpid]['profile'][:,1][dos[comp][mpid]['rows']],s=18,edgecolors='k',facecolors='none')
        if plot is True:
            plt.xlabel('Energy [eV]')
            plt.ylabel('Density of states [arb.u.]')
            plt.legend()
            plt.title(rf'{comp} DOS model, valence band region')
            plt.show()
    return dos