# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 18:06:07 2022

@author: James Nohl
"""

import pysehi as ps
import numpy as np
import math
from scipy.ndimage import gaussian_filter as gf
import tifffile as tf

test_img_path = r"G:\My Drive\Data\Collaboration\Processed\Loughborough\NMC\622\cathode\221209\BC1\BC1_avg_img.tif"
img = ps.load_single_file(test_img_path)[0]

def circle_mask(img, radius):
    result = np.zeros(img.shape,np.float64)
    centerX = (img.shape[0])/2
    centerY = (img.shape[1])/2
    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            if math.sqrt((m-centerX)**2+(n-centerY)**2) < radius:
                result[m,n] = img[m,n]
    return result

def degree2rad(degrees):
    return degrees*np.pi/180

def segment_mask(img, radius=150, start_angle=177, end_angle=183):#, seg=False):
    cy,cx = np.array(img.shape)/2
    sta = degree2rad(start_angle)
    ea = degree2rad(end_angle)
    t = np.linspace(sta, ea, end_angle*3)
    x = cx + radius*np.cos(t)
    y = cy +radius*np.sin(t)
    path = (x[0],y[0])
    for xc, yc in zip(x[1:], y[1:]):
        path = np.vstack((path,np.array([xc,yc])))
    return path
    #else: #disk sector
    #    return np.vstack((path,np.array([cx,cy]))) #sector

def relabel(data):
    if type(data) is str:
        data = ps.data(data)
    labels = []
    for i,page in enumerate(data.stack_meta):
        labels.append(rf'TLD_Mirror{i+1}_'+str(data.stack_meta[page]['TLD']['Mirror'])+'.tif')
    pixel_width_um = data.stack_meta['img1']['Scan']['PixelWidth']*1e6
    save_path = data.folder
    tf.imwrite(rf'{save_path}\{data.name}_stack.tif',
               data.stack, dtype=data.dtype_info.dtype, photometric='minisblack', imagej=True,
               resolution=(1./pixel_width_um, 1./pixel_width_um), metadata={'spacing':1, 'unit': 'um', 'axes':'ZYX', 'Labels':labels}) #make numpy array into multi page OME-TIF format (Bio - formats)
    