# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:07:50 2022

@author: James Nohl
"""

import os
import glob
import tifffile as tf
import json
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from scipy.ndimage import uniform_filter as UF
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.cm as comap
import matplotlib.colors as colors
import smooth

def gmm_seg(img, max_comp=10, n_components=None):
    if n_components is None:
        img_uf = smooth.uniform(img)
        img1=img_uf.reshape((img.size,1))
        bics = []
        for i in range(max_comp): # test the AIC/BIC metric between 1 and 10 components
            gmm = GMM(n_components = i+1, covariance_type = 'full')
            labels = gmm.fit(img1).predict(img1)
            bic = gmm.bic(img1)
            bics.append(bic)
        n_components = opt_comp_bics(bics)
    
    n_x = np.arange(np.min(img),np.max(img)+1,1)

    gmm_model = GMM(n_components).fit(img1)
    gmm_labels = gmm_model.predict(img1)
    gauss_mixt = np.array([p * norm.pdf(n_x, mu, sd) for mu, sd, p in zip(gmm_model.means_.flatten(), np.sqrt(gmm_model.covariances_.flatten()), gmm_model.weights_)])
    gauss_mixt_t = np.sum(gauss_mixt, axis = 0)
    
    res = {}
    res['gmm']={}
    res['gmm']['mu']=gmm.means_.flatten()
    res['gmm']['sd']=gmm.covariances_.flatten()
    res['gmm']['p']=gmm.weights_
    res['gmm']['pdf']=gauss_mixt
    res['gmm']['pdf_sum']=gauss_mixt_t
    
    color = comap.viridis(np.linspace(0,1,len(gauss_mixt)))
    y, x, _ = plt.hist(img1, bins=np.max(img1), rwidth=0.8, alpha=0.5, color='k')
    for i, (g, c) in enumerate(zip(res['gmm']['pdf'],color)):
        plt.plot(n_x, (g/max(res['gmm']['pdf_sum']))*np.max(y), label = f'Gaussian {i}',color=c)
        plt.plot(n_x,res['gmm']['pdf_sum'])
    plt.show()
    
    orig = img.shape
    segmented = gmm_labels.reshape(orig[0],orig[1])
    n = np.max(segmented)+1
    
    from_list = colors.LinearSegmentedColormap.from_list
    cm = from_list(None, plt.cm.Set1(range(0,n)), n)
    plt.imshow(segmented, cmap=cm)
    plt.clim(-0.5, n-0.5)
    cb = plt.colorbar(ticks=range(0,n), label='Group')
    cb.ax.tick_params(length=0)

def opt_comp_bics(bics):
    bics_change = max(bics)-min(bics)
    num_count = []
    for i in range(len(bics)):
        if i > 0:
            change = ((bics[i-1] - bics[i])/bics_change)*100
            if change > 3 and bics[i] >= 0:
                num_count.append(i+1)
    opt_bic = num_count[-1]
    return opt_bic

def compute_contour(thresh, size_pix=0):
    from imutils import contours
    from skimage import measure
    import imutils
    import cv2
    """
    This method computes the contour of the particles available in the EM images

    :param thresh: input binary image
    :return: contours of the particles
    """
    # Label connected regions of the input image.
    # Two pixels are connected when they are neighbors and have the same value.
    # background pixels are labeled as 0
    labels = measure.label(thresh, background=0)

    # create a mask image of the size of input image
    mask = np.zeros(thresh.shape, dtype="uint8")

    # loop over the unique labels of particles
    for label in np.unique(labels):
        # check for the background pixel
        if label == 0:
            continue

        # otherwise, construct the label mask and count the number of pixels
        label_mask = np.zeros(thresh.shape, dtype="uint8")
        label_mask[labels == label] = 255

        # check the number of pixels in that particular label
        num_pixels = cv2.countNonZero(label_mask)

        # if the number of pixels in that particular particle  is sufficiently large then keep it
        # and that is defined by user"
        if num_pixels > size_pix:
            mask = cv2.add(mask, label_mask)

    # Finds contours of particles from the binary mask image. The mode is set to
    # retrieves only the extreme outer contours an uses simple chain approximation to store the contour points
    contours_image = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res={}
    for i in range(len(contours_image[0])):
        res[i]={}
        res[i]['roi_path']=contours_image[0][i][:,0,:]

    return res

def plot_res(segmented_8bit,res):
    plt.imshow(segmented_8bit,cmap='gray')
    for i in range(len(res)):
        plt.plot(res[i][:,0,0],res[i][:,0,1])
    plt.show()