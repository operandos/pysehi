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
import matplotlib.pyplot as plt
from pyvsnr import VSNR
import time

test_img_path = r"G:\My Drive\Data\Collaboration\Processed\Loughborough\NMC\622\cathode\221209\BC1\BC1_avg_img.tif"
img = ps.load_single_file(test_img_path)[0]

def curtains_removal_vsnr(img, maxit=15):
    if len(img.shape) == 3:
        z,y,x = img.shape
        stack = np.asarray(img, dtype='float64')
    if len(img.shape) == 2:
        y,x = img.shape
        img=np.asarray(img, dtype'float64')
    if y%2 != 0:
        y-=1
    if x%2 != 0:
        x-=1
    
    # vsnr object creation
    vsnr = VSNR([y,x])
    
    # add filter (at least one !)
    vsnr.add_filter(alpha=1e-2, name='gabor', sigma=(1, 30), theta=358)
    
    # vsnr initialization
    vsnr.initialize()

    # image processing
    stack_corr = []
    t0 = time.process_time()
    for i,page in enumerate(stack):
        t1 = time.process_time()
        img_corr = vsnr.eval(page[:y,:x], maxit=maxit, cvg_threshold=1e-4)
        img_corr_uint8 = np.asarray(img_corr, dtype='uint8')
        stack_corr.append(img_corr_uint8)
        #print("CPU/GPU running time :", time.process_time() - t1, "\t remaining: ", len(stack)-i-1)
    print("DONE!..........elapsed time :", time.process_time() - t0)
    stack_corr = np.asarray(stack_corr)
    
    # plotting
    fig0 = plt.figure(figsize=(12, 6))
    fig0.sfn = "ex_fib_sem"
    plt.subplot(121)
    plt.title("Original")
    plt.imshow(np.average(stack,0), cmap='gray')
    plt.subplot(122)
    plt.title("Corrected")
    plt.imshow(np.average(stack_corr,0), cmap='gray')
    plt.tight_layout()

def curtains_correction(img, radius=200, start_angle=179, end_angle=181):
    amp = cheatfft(img)
    rois = segment_mask(img, radius, start_angle, end_angle)
    mask = rois[0]['img_mask']
    mask_app = np.ones((amp.shape))
    mask_app = np.where(mask == False, mask_app,0)
    mask_app = gf(mask_app,0.5)
    amp_mask = amp*mask_app
    res_amp_mask = np.abs(np.fft.ifft2(np.fft.ifftshift(amp_mask)))
    plt.imshow(res_amp_mask,cmap='gray')
    plt.show()

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

def segment_mask(img, radius, start_angle, end_angle):
    cy,cx = np.array(img.shape)/2
    sta = degree2rad(start_angle)
    ea = degree2rad(end_angle)
    t = np.linspace(sta, ea, end_angle*3)
    x = cx + radius*np.cos(t)
    y = cy +radius*np.sin(t)
    path = (x[0],y[0])
    for xc, yc in zip(x[1:], y[1:]):
        path = np.vstack((path,np.array([xc,yc])))
    sect = np.vstack((path,np.array([cx,cy]))) #sector
    rois = ps.roi_masks(img, sect)
    rois[0]['img_mask'] = np.ma.mask_or(rois[0]['img_mask'],np.flip(rois[0]['img_mask']))
    return rois

def cheatfft(img, plot=False):
    amp = (np.fft.fftshift(np.fft.fft2(img)))
    return amp
    if plot:
        plt.imshow(np.log(np.abs(amp)))
        plt.show()

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
    