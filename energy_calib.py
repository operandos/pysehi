# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 10:38:49 2022

@author: James Nohl
"""

import os
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pysehi as ps

path_to_files = r"G:\My Drive\Data\Collaboration\Processed\Loughborough\Si\221208\Si_calib_25pA_-3_MV"
def energy_calib_coeffs(path_to_files):
    data = ps.list_files(path_to_files, condition_true=['Si'], load_data=True)
    
    if not os.path.exists(rf'{path_to_files}\calibration_outputs'):
        os.mkdir(rf'{path_to_files}\calibration_outputs')
    
    biases=[]
    for name in data:
        bias_V = data[name]['data'].stack_meta['img1']['Beam']['FineStageBias']
        biases.append(bias_V)
    biases = natsorted(biases)
    
    n = rf"{list(data.keys())[0].split('_')[0]}_{list(data.keys())[0].split('_')[1]}"
    
    color = cm.viridis(np.linspace(0,1,len(data)))
    grx=[]
    inty=[]
    for b,c in zip(biases, color):
        name = rf'{n}_{b}V'
        bias_V = data[name]['data'].stack_meta['img1']['Beam']['FineStageBias']
        x = ps.MV(data[name]['data'].stack_meta)
        x_interp = np.linspace(x[0],x[-1],300)
        y_interp = np.interp(x_interp,x[::-1],ps.zpro(data[name]['data'].stack)[::-1])
        maxX = np.argmax(np.gradient(y_interp))
        grx.append(x_interp[maxX])
        plt.plot(x, ps.zpro(data[name]['data'].stack), label = f'V = {bias_V}',color=c)
        inty.append(bias_V)
        plt.plot(x_interp[maxX],y_interp[maxX], 'kx')
    plt.xlabel('Mirror voltage')
    plt.tick_params(axis='both', which='both', bottom=True, top=False, labelbottom=True, right=False, left=False, labelleft=False)
    plt.ylabel('Intensity')
    plt.plot(x_interp[maxX],y_interp[maxX], 'kx', label = 'max grad')
    plt.legend(loc='lower right')
    plt.gca().invert_xaxis()
    plt.savefig(rf'{path_to_files}\calibration_outputs\S-curve_with_max_grad_pts.png',dpi=400,transparent=True)
    plt.show()
    
    E = np.array(biases)*-1
    
    plt.plot(grx,E, 'ko')
    plt.show()
    order = int(input('Intput polyfit order:   '))
    
    coeffs = np.polyfit(grx, E, order)
    np.savetxt(rf'{path_to_files}\{np.datetime64("today")}_calib_coeffs.csv',coeffs) #put this file at the top of the directory
    
    plt.plot(grx,E, 'ko')
    plt.plot(grx,np.polyval(coeffs, grx), 'k--')
    #plt.xlim(left=-0.5)
    plt.xlabel('MV at max grad [V]')
    plt.ylabel('Energy shift equivalent $\it{E}$ [eV]')
    plt.xlim()
    #plt.annotate('$\it{E}$[eV] = '+f'{round(m1,4)}[eV/V]'+r' $\times$ MV[V] + '+f'{round(b1,3)}[eV]', [min(grx)+0.5,max(m1*np.array(grx)+b1)], fontsize=12)
    plt.annotate(coeffs, [min(grx)+0.5,max(np.polyval(coeffs, grx))], fontsize=10)
    #print(f'factor from grad = {m1}')
    plt.savefig(rf'{path_to_files}\calibration_outputs\factor_plot.png',dpi=400,transparent=True)
    plt.show()
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for b, c in zip(biases,color):
        name = rf'{n}_{b}V'
        bias_V = data[name]['data'].stack_meta['img1']['Beam']['FineStageBias']
        ax1.plot(ps.MV(data[name]['data'].stack_meta), ps.zpro(data[name]['data'].stack), label = f'V = {bias_V}',color=c)
    ax1.set_xlabel('Mirror voltage [V]')
    ax1.tick_params(axis='both', which='both', bottom=True, top=False, labelbottom=True, right=False, left=False, labelleft=False)
    ax1.set_ylabel('Intensity')
    ax1.legend(loc='lower right')
    plt.gca().invert_xaxis()
    def tick_f(X):
        xt = np.polyval(coeffs,X) + 6
        return ["%.2f" % z for z in xt]
    tick_loc = np.arange(np.min(ps.MV(data[name]['data'].stack_meta)),np.max(ps.MV(data[name]['data'].stack_meta))+1,5)[::-1]
    ax2 = ax1.twiny()
    ax2.set_xticks(tick_loc)
    ax2.set_xticklabels(tick_f(tick_loc))
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xlabel('Energy equivalent ($\it{E}$) [eV] (+ 6 shift)')
    plt.savefig(rf'{path_to_files}\calibration_outputs\S-curve_with_eV.png',dpi=400,transparent=True)
    plt.show()
    return coeffs

def energy_calib_shift(carbon_data):
    
    