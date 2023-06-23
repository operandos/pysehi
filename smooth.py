# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:23:53 2022

@author: James Nohl
"""
import numpy as np
from scipy import signal
from scipy.stats import multivariate_normal as mn
from scipy.stats import mode
from scipy.stats import mode
from scipy.ndimage import uniform_filter as uf

def gauss(stack,xy_smooth=1,z_smooth=None):
    if isinstance(stack,np.ndarray):
        if z_smooth is None:
            z_smooth = xy_smooth
        in_sigma = [xy_smooth,xy_smooth,z_smooth]
        s = np.diag(in_sigma)
        maxSize = 2*np.ceil(2*np.max(in_sigma))+1
        xx = np.arange(-maxSize,maxSize+1,1)   # coordinate arrays -- make sure they contain 0!
        yy = np.arange(-maxSize,maxSize+1,1)
        zz = np.arange(-maxSize,maxSize+1,1)
        [gaussGridX, gaussGridY, gaussGridZ] = np.meshgrid(xx,yy,zz)
        coord=np.array([gaussGridX.flatten(), gaussGridZ.flatten(), gaussGridY.flatten()]).T
        mu = np.mean(coord, axis =0)
        outsize = gaussGridX.shape
        p = mn.pdf(coord, mu, s)
        nonIsoGauss = p.reshape(outsize)
        filtered = signal.convolve(stack, nonIsoGauss, mode="same")
        stack_GF = filtered.transpose(1,2,0)
        stack_GF = np.gradient(stack_GF, axis=2)
        return stack_GF
    else:
        print('stack is not image stack')
        return

def uniform(stack,size=3):
    if isinstance(stack,np.ndarray):
        stack_UF = uf(stack,size)
        return stack_UF
    else:
        print('stack is not image stack')

def mov_av(y, width):
    box = np.ones(width)/width
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth