# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 10:55:03 2022

@author: James Nohl
"""

import numpy as np
import tifffile as tf
from skimage import transform
import json
import regex
import glob
import os
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from skimage import img_as_ubyte

def sortKey1(file):
    m = regex.search("(?<=TLD_Mirror\d+_)[-+]?\d*([.](?!tif)\d*)*", file) # searches for the mirrorV
    if m is None:
        m = regex.search("(?<=TLD_Mirror_\d+_)[-+]?\d*([.](?!tif)\d*)*", file) # searches for the mirrorV in oxford data
    return float(m.group())
def sortKey2(file):
    m = regex.search("\d+", file)
    return int(m.group())

direct = input('paste path to folder with processed stacks to align:    ')
direct = direct.replace('"','')

alignment = input('alignment metafile:   ')
alignment=alignment.replace('"','')
#alignment = r"G:\My Drive\Data\James\Processed\HOPG\220324\Pre\Tilt2\AG2\Metadata\AG2-stack_metadata.json"

with open(alignment) as file:
        align_metadata = json.load(file)
PixelWidthSI = align_metadata['img1']['EScan']['PixelWidth']
pixelWidth = PixelWidthSI*1e6
"""
if 'crop' in align_metadata['img1']['Processing']:
    cropX0, cropX1 = align_metadata['img1']['Processing']['crop']['x0'],align_metadata['img1']['Processing']['crop']['x1']
    cropY0, cropY1 = align_metadata['img1']['Processing']['crop']['y0'],align_metadata['img1']['Processing']['crop']['y1']
else:
    tl=[]
    for page in align_metadata:
        tl.append([align_metadata[page]['Processing']['transformation']['x'],align_metadata[page]['Processing']['transformation']['y']])
    cropX0, cropX1 = np.min()
"""
shift_list = []
for img in align_metadata:
    shiftX,shiftY = align_metadata[img]['Processing']['transformation']['x'],align_metadata[img]['Processing']['transformation']['y']
    shift_list.append([shiftX*PixelWidthSI,shiftY*PixelWidthSI])
shift_list = np.array(shift_list)

#plt.scatter(np.linspace(0,len(shift_list[:,0]),len(shift_list[:,0])),shift_list[:,0], label='x shift')
#plt.scatter(np.linspace(0,len(shift_list[:,1]),len(shift_list[:,1])),shift_list[:,1], label='y shift')
#plt.legend()
#plt.show()

valx_list=[]
for i in np.arange(0,len(shift_list[:,0])-1,1):
    i=int(i)
    val = shift_list[i,0]-shift_list[i+1,0]
    valx_list.append(val)
valy_list=[]
for i in np.arange(0,len(shift_list[:,1])-1,1):
    i=int(i)
    val = shift_list[i,1]-shift_list[i+1,1]
    valy_list.append(val)

plt.scatter(np.arange(0,len(shift_list[:,0]),1),shift_list[:,0], label='x shift')
plt.scatter(np.arange(0,len(shift_list[:,1]),1),shift_list[:,1], label='y shift')
plt.show()

#lim = np.argmax(np.abs(valy_list))+1
lim = int(input('input the slice where extrapolation should begin:   '))

my,cy = np.polyfit(np.arange(lim,len(shift_list[:,1]),1),shift_list[lim:,1],1)
mx,cx = np.polyfit(np.arange(lim,len(shift_list[:,0]),1),shift_list[lim:,0],1)

plt.scatter(np.arange(0,len(shift_list[:,0]),1),shift_list[:,0], label='x shift')
plt.plot(np.arange(0,len(shift_list[:,0]),1),mx*np.arange(0,len(shift_list[:,0]),1)+cx,'k',label='x fit',linestyle='--')
plt.scatter(np.arange(0,len(shift_list[:,1]),1),shift_list[:,1], label='y shift')
plt.plot(np.arange(0,len(shift_list[:,1]),1),my*np.arange(0,len(shift_list[:,1]),1)+cy,'k',label='y fit')
plt.xlabel('slice no.')
plt.ylabel('shift [m]')
plt.legend()
plt.savefig(rf'{direct}\align_by_extrap_plot.png')
plt.show()

for folder in os.listdir(direct):
    if os.path.exists(rf'{direct}\{folder}\Metadata') == True and not '.png' in folder:# and not alignment.split(r'\Metadata')[0] in folder:
        print(f'processing..............{folder}')
        if os.path.exists(rf'{direct}\{folder}\Metadata\{folder}-stack_metadata.json'):
            f = rf'{direct}\{folder}\Metadata\{folder}-stack_metadata.json'
        if os.path.exists(rf'{direct}\{folder}\Metadata\{folder}_stack_meta.json'):
            f = rf'{direct}\{folder}\Metadata\{folder}_stack_meta.json'
        with open(f) as file:
            stack_metadata = json.load(file)
        file.close()
        PixelWidthSI_1 = stack_metadata['img1']['EScan']['PixelWidth']
        resX,resY = stack_metadata['img1']['Image']['ResolutionX'],stack_metadata['img1']['Image']['ResolutionY']
        root = direct.replace('Processed','Raw')
        files = glob.glob(rf'{root}\{folder}\*.tif')
        sortedFiles = []
        for file in sorted(files, key = sortKey1, reverse=True):
            sortedFiles.append(file)
    
        refImg = tf.imread(sortedFiles[-1])
        dtype = str(refImg.dtype)
        
        regImgList = []
        labels=[]
        shift_list_1 = []
        for i,file in enumerate(sortedFiles):
            shiftX,shiftY = mx*i+cx,my*i+cy #shift required [m]
            shiftX,shiftY = shiftX/PixelWidthSI_1,shiftY/PixelWidthSI_1 #convert from m to pixels
            shift_list_1.append([shiftX,shiftY])
            rawImg = tf.imread(file)
            tform = transform.EuclideanTransform(translation = (shiftX,shiftY))
            regImg = transform.warp(rawImg, tform)
            regImg = img_as_ubyte(regImg)
            #plt.imshow(regImg)  #for testing
            plt.show()
            regImgList.append(regImg)
            labels.append(os.path.split(file)[1])
        regImgArr = np.array(regImgList)
        
        xMax, yMax = np.ceil(np.max(shift_list_1,axis=0))
        xMin, yMin = np.floor(np.min(shift_list_1,axis=0))
        regImgArr_preserve = regImgArr
        
        if xMin > 0:
            regImgArr = regImgArr[:,:,int(xMax-xMin):int(resX-xMax)]
            if yMax > 0:
                if yMin < 0:
                    regImgArr = regImgArr[:,int(0-yMin):int(resY-yMax),:]
                else:
                    regImgArr = regImgArr[:,int(yMin):int(resY-yMax),:]
            if yMax < 0:
                if abs(yMin) > abs(yMax):
                    regImgArr = regImgArr[:,int(0-yMin):,:]                
        else:
            if yMax > 0:
                if yMin < 0:
                    regImgArr = regImgArr[:,int(0-yMin):int(resY-yMax),:]
                else:
                    regImgArr = regImgArr[:,int(yMin):int(resY-yMax),:]
            if yMax < 0:
                if abs(yMin) > abs(yMax):
                    regImgArr = regImgArr[:,int(0-yMin):,:]                
            regImgArr = regImgArr[:,:,int(0-xMin):int(resX-xMax)]
            
        
        AVG = np.array(np.mean(regImgArr, axis=0))
        plt.figure()
        plt.imshow(AVG, cmap='gray')
        plt.axis('off')
        scalebar = ScaleBar(dx=PixelWidthSI_1, length_fraction=0.3, location='lower right', border_pad=0.5, color='w', box_color='k', box_alpha=0.5, font_properties={'size':'15'})
        plt.gca().add_artist(scalebar)
        bbox_inches = 0
        plt.savefig(rf"{direct}\{folder}\{folder}_AVG-scaled_aligned.png", dpi=300, bbox_inches='tight',pad_inches=0)
        plt.show()
        
        tf.imwrite(rf"{direct}\{folder}\{folder}-average-img_aligned.tif",np.array(np.mean(regImgArr, axis=0),dtype='uint8'),dtype = 'uint8', photometric='minisblack', imagej=True,
                     resolution=(1./(PixelWidthSI_1*1e6), 1./(PixelWidthSI_1*1e6)), metadata={'unit': 'um', 'axes':'YX'})
        tf.imwrite(rf"{direct}\{folder}\{folder}-match-crop-stack_aligned.tif", regImgArr, dtype = 'uint8', photometric='minisblack', imagej=True,
                     resolution=(1./(PixelWidthSI_1*1e6), 1./(PixelWidthSI_1*1e6)), metadata={'spacing':np.arange(0,len(sortedFiles),1), 'unit': 'um', 'axes':'ZYX', 'Labels':labels}) #make numpy array into multi page OME-TIF format (Bio - formats)

    if 'stack.tif' in folder:
        print(f'processing..............{direct}')
        with open(rf'{direct}\Metadata\{os.path.split(direct)[1]}_stack_meta.json') as file:
            stack_metadata = json.load(file)
        file.close()
        PixelWidthSI_1 = stack_metadata['img1']['EScan']['PixelWidth']
        resX,resY = stack_metadata['img1']['Image']['ResolutionX'],stack_metadata['img1']['Image']['ResolutionY']
        root = direct.replace('Processed','Raw')
        files = glob.glob(rf'{root}\*.tif')
        sortedFiles = []
        if 'Helios' in stack_metadata['img1']['System']['SystemType']:
            for file in sorted(files, key = sortKey1, reverse=True):
                sortedFiles.append(file)
        if 'Nova' in stack_metadata['img1']['System']['SystemType']:
            for file in sorted(files, key = sortKey2, reverse=True):
                sortedFiles.append(file)
    
        refImg = tf.imread(sortedFiles[-1])
        dtype = str(refImg.dtype)
        
        regImgList = []
        labels=[]
        shift_list_1 = []
        for i,file in enumerate(sortedFiles):
            shiftX,shiftY = mx*i+cx,my*i+cy #shift required [m]
            shiftX,shiftY = shiftX/PixelWidthSI_1,shiftY/PixelWidthSI_1 #convert from m to pixels
            shift_list_1.append([shiftX,shiftY])
            rawImg = tf.imread(file)
            tform = transform.EuclideanTransform(translation = (shiftX,shiftY))
            regImg = transform.warp(rawImg, tform)
            regImg = img_as_ubyte(regImg)
            regImgList.append(regImg)
            labels.append(os.path.split(file)[1])
        regImgArr = np.array(regImgList)
        
        xMax, yMax = np.ceil(np.max(shift_list_1,axis=0))
        xMin, yMin = np.floor(np.min(shift_list_1,axis=0))
        
        if xMin > 0:
            regImgArr = regImgArr[:,:,int(xMax-xMin):int(resX-xMax)]
            if yMax > 0:
                regImgArr = regImgArr[:,int(0-yMin):int(resY-yMax),:]
            if yMax < 0:
                if abs(yMin) > abs(yMax):
                    regImgArr = regImgArr[:,int(0-yMin):,:]                
        else:
            regImgArr = regImgArr[:,int(0-yMin):int(resY-yMax),int(0-xMin):int(resX-xMax)]
        
        AVG = np.array(np.mean(regImgArr, axis=0))
        plt.figure()
        plt.imshow(AVG, cmap='gray')
        plt.axis('off')
        scalebar = ScaleBar(dx=PixelWidthSI_1, length_fraction=0.3, location='lower right', border_pad=0.5, color='w', box_color='k', box_alpha=0.5, font_properties={'size':'15'})
        plt.gca().add_artist(scalebar)
        bbox_inches = 0
        plt.savefig(rf"{direct}\{os.path.split(direct)[1]}_AVG-scaled_aligned.png", dpi=300, bbox_inches='tight',pad_inches=0)
        plt.show()
        
        tf.imwrite(rf"{direct}\{os.path.split(direct)[1]}-average-img_aligned.tif",np.array(np.mean(regImgArr, axis=0),dtype='uint8'),dtype = 'uint8', photometric='minisblack', imagej=True,
                     resolution=(1./(PixelWidthSI_1*1e6), 1./(PixelWidthSI_1*1e6)), metadata={'unit': 'um', 'axes':'YX'})
        tf.imwrite(rf"{direct}\{os.path.split(direct)[1]}-match-crop-stack_aligned.tif", regImgArr, dtype = 'uint8', photometric='minisblack', imagej=True,
                     resolution=(1./(PixelWidthSI_1*1e6), 1./(PixelWidthSI_1*1e6)), metadata={'spacing':np.arange(0,len(sortedFiles),1), 'unit': 'um', 'axes':'ZYX', 'Labels':labels}) #make numpy array into multi page OME-TIF format (Bio - formats)
print('######   Done!   ######')