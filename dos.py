# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:07:01 2022

@author: James Nohl
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import json

def load_dos_npy(compounds:list or str, path_to_dos_lib:str,xlim:float=7, plot:bool=False):
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

def compound_contains(comp:str, path_to_dos_lib:str, condition_false:str or list=None):
    files = glob.glob(rf"{path_to_dos_lib}\*{comp}*")
    comps=[]
    for f in files:
        if condition_false is not None:
            if any(c in f for c in condition_false):
                continue
            else:
                comps.append(os.path.split(f)[1].split('.')[0])
        else:
            comps.append(os.path.split(f)[1].split('.')[0])
    return comps

def norm(series, minmax=True):
    if minmax is True:
        series_norm = (series-np.min(series))/(np.max(series)-np.min(series))
    else:
        series_norm = series/np.max(series)
    return series_norm

def plot_dos(dos, show=True, normalise=False):
    for comp in dos:
        for mpid in dos[comp]:
            x = dos[comp][mpid]['profile'][:,0][dos[comp][mpid]['rows']]
            if normalise is True:
                y = norm(dos[comp][mpid]['profile'][:,1][dos[comp][mpid]['rows']])
            else:
                y = dos[comp][mpid]['profile'][:,1][dos[comp][mpid]['rows']]
            plt.plot(x,y,label=mpid)
            plt.scatter(x,y,s=18,edgecolors='k',facecolors='none')
        if show is True:
            plt.xlabel('Energy [eV]')
            plt.ylabel('Density of states [arb.u.]')
            plt.title(rf'{comp} DOS model, valence band region')
            plt.legend()
            plt.show()
            
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)