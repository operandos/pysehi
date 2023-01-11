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

import sys
sys.path.append(r"G:\My Drive\Code\Exports\GitHub\pysehi")
import pysehi as ps
import sys
sys.path.append(r"G:\My Drive\Code\Exports\GitHub\private")
import sespec as spc

def energy_calib_coeffs(path_to_Si, HOPG_dat=None, HOPG_ref=None, csv_output=None):
    data = ps.list_files(path_to_Si, condition_true=['Si'],condition_false=['_1'], load_data=True)
    
    if csv_output is None:
        csv_output = path_to_Si
    if not os.path.exists(rf'{path_to_Si}\calibration_outputs'):
        os.mkdir(rf'{path_to_Si}\calibration_outputs')
    
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
    plt.savefig(rf'{path_to_Si}\calibration_outputs\S-curve_with_max_grad_pts.png',dpi=400,transparent=True)
    plt.show()
    
    E = np.array(biases)*-1
    
    plt.plot(grx,E, 'ko')
    plt.show()
    order = int(input('Intput polyfit order:   '))
    
    coeffs = np.polyfit(grx, E, order)
    plt.plot(grx,E, 'ko')
    plt.plot(grx,np.polyval(coeffs, grx), 'k--')
    #plt.xlim(left=-0.5)
    plt.xlabel('MV at max grad [V]')
    plt.ylabel('Energy shift equivalent $\it{E}$ [eV]')
    plt.xlim()
    plt.annotate(coeffs, [min(grx)+0.5,max(np.polyval(coeffs, grx))], fontsize=10)
    #print(f'factor from grad = {m1}')
    plt.title('pre-shift coefficients')
    plt.savefig(rf'{path_to_Si}\calibration_outputs\factor_plot.png',dpi=400,transparent=True)
    plt.show()
    
    if HOPG_dat is None:
        np.savetxt(rf'{csv_output}\{np.datetime64("today")}_calibration.csv',coeffs)
    if HOPG_dat is not None:
        HOPG_dat = ps.data(HOPG_dat)
        HOPG_ref = ps.data(HOPG_ref)
        eV_pre = np.array(np.polyval(coeffs, ps.MV(HOPG_dat.stack_meta)))
        max_HOPG_dat = eV_pre[np.argmax(HOPG_dat.spec())]
        max_HOPG_ref = HOPG_ref.eV[np.argmax(HOPG_ref.spec())]
        shift = max_HOPG_ref-max_HOPG_dat
        coeffs[-1]=coeffs[-1]+shift
        eV_post = np.array(np.polyval(coeffs, ps.MV(HOPG_dat.stack_meta)))
        np.savetxt(rf'{csv_output}\{np.datetime64("today")}_calibration.csv',coeffs)
        
        fig = plt.figure(figsize=(8.5,4))
        fig.suptitle('Reference HOPG plots',weight='bold')
        plt.subplot(1,2,1)
        plt.plot(HOPG_ref.eV, ps.norm(HOPG_ref.spec()),label='ref')
        plt.plot(eV_pre, ps.norm(HOPG_dat.spec()),label='dat')
        plt.title('Pre shift')
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(HOPG_ref.eV, ps.norm(HOPG_ref.spec()),label='ref')
        plt.plot(eV_post, ps.norm(HOPG_dat.spec()),label='dat')
        plt.title(rf'Post shift (+{round(shift,8)})')
        plt.legend()
        plt.savefig(rf'{path_to_Si}\calibration_outputs\shift_plots.png',dpi=400,transparent=True)
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
    plt.savefig(rf'{path_to_Si}\calibration_outputs\S-curve_with_eV.png',dpi=400,transparent=True)
    plt.show()
    
    return coeffs