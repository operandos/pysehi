# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 12:35:00 2022

@author: James Nohl
"""
import glob
import tifffile as tf
import regex
import numpy as np
import matplotlib.pyplot as plt

def load(folder, factor_helios = -0.39866666666666667, factor_nova = 2.84, corr = 6):
    if 'Processed' in folder:
        processed = True
    if 'Raw' in folder:
        processed = False
        files = glob.glob(rf'{folder}\*.tif')
        ana_voltage = []
        for file in files:
            metadata = load_single_file(file, load_img = False)
            if 'Helios' in metadata['System']['SystemType']:
                sys = True
                analyser = 'Mirror'
            if 'Nova' in metadata['System']['SystemType']:
                sys = False
                analyser = 'Deflector'
            ana_voltage.append(metadata['TLD'][analyser])
        files_sorted = [files for (ana_voltage,files) in sorted(zip(ana_voltage,files), key=lambda pair: pair[0], reverse=sys)]
        imgs = []
        stack_meta = {}
        ana_voltage = []
        for i,file in enumerate(files_sorted):
            img, metadata = load_single_file(file)
            stack_meta[f'img{i}'] = metadata
            imgs.append(img)
            ana_voltage.append(metadata['TLD'][analyser])
        stack = np.dstack(imgs)
        if sys==True:
            eV = np.array(ana_voltage)*factor_helios+corr
        if sys==False:
            eV = np.array((ana_voltage*factor_nova))
        return stack, stack_meta, eV
    
def load_single_file(file, load_img = True):
    with tf.TiffFile(file) as tif:
        metadata = tif.fei_metadata
    tif.close()
    if load_img == True:
        img = tf.imread(file)
        return img, metadata
    else:
        return metadata

def conversion(stack_meta, factor, corr):
    MV = []
    for i in stack_meta:
        stack_meta[f'img{i}']['']
    eV = (MV*factor)+corr
    return eV

def plot_axes():
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.it'] = 'sans:italic:bold'
    plt.rcParams['mathtext.bf'] = 'sans:bold'
    plt.xlabel('Energy, $\mathit{E}$ [eV]',weight='bold')
    plt.ylabel('Emission intensity norm. [arb.u.]',weight='bold')
        
class data:
    factor = -0.39866666666666667
    corr = 6
    def __init__(self, folder):
        self.folder = folder
        self.stack, self.stack_meta, self.eV = load(folder)
        self.dims = self.stack.shape   
    def zpro(self):
        zpro = []
        for i in np.arange(0,self.dims[2],1):
            zpro.append(np.mean(self.stack[:,:,i]))
        return zpro
    def spec(self):
        spec = np.gradient(data.zpro(self))
        return spec
    def plot_img(self, avg_img = True, fin_img = False, ):
        if avg_img == True and fin_img == False:
            plt.imshow(np.mean(self.stack,axis=2),cmap='gray')
            plt.axis('off')
            plt.show()
        if fin_img == True:
            plt.imshow(self.stack[:,:,-1],cmap='gray')
            plt.axis('off')
            plt.show()
    def plot_spec(self):
        plt.plot(self.eV, data.spec(self))
        plot_axes()
        plt.show()
    def plot_zpro(self):
        plt.plot(self.eV, data.zpro(self))
        plot_axes()
        plt.show()
    