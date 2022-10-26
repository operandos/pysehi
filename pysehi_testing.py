# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 12:35:00 2022

@author: James Nohl
"""
import os
import glob
import tifffile as tf
import regex
import json
import numpy as np
import scipy.ndimage as scnd
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.cm as cm
from matplotlib import colors
from cv2 import matchTemplate as match_template
from cv2 import minMaxLoc as min_max_loc
from cv2 import TM_CCOEFF_NORMED
from skimage import transform
from skimage.registration import phase_cross_correlation as pcc
from skimage import img_as_ubyte
from skimage import img_as_float
import read_roi

folder_nova = r"G:\My Drive\Data\James\Raw\HOPG\210601\G4_Data"
folder_helios = r"G:\My Drive\Data\James\Raw\HOPG\220926\AB2"
folder_AC = r"G:\My Drive\Data\James\Raw\Pristine\Powder\NMC\622\BSAF\221006_1\BG1"
folder_processed = r"G:\My Drive\Data\James\Processed\Graphene\220824\Gr1\AC1_AC"

def process_files(path_to_files, AC:bool=True, condition_true:list=None, condition_false:list=None):
    if 'Raw' in path_to_files:
        data_files = list_files(path_to_files, condition_true, condition_false)
        for name in data_files:
            root = data_files[name]['Raw_path']
            if os.path.exists(root.replace('Raw','Processed')) is False:
                if '_R' not in root:
                    print(rf'loading.........{root}')
                    data_files[name]['Processed_path'] = path_to_files.replace('Raw','Processed')
                    data(root, AC=AC).save_data()
                    data_files[name]['stack_meta'] = data(root).stack_meta
                    print(rf'{root}.........processed!')
        return data_files
    else:
        print(r'already processed files associated with Raw_path')

def list_files(path_to_files, condition_true:list=None, condition_false:list=None, load_data=False):
    """
    List raw files for processing or processed files for analysis
    Requires data file structure to be located using the scheme ...\Raw\Material\YYMMDD\Subclass\folder

    Parameters
    ----------
    path_to_files : str
        path to raw or processed files from a certain date
    conditions : list
        list of strings that must be included in root for root to be added to list of files
    load_data : bool
        if True, will add asociated .tif stack to dict

    Returns
    -------
    data_files : dict
        dictionary of data files with keys such as date, material...

    """
    data_files={}
    for root, _, file_list in os.walk(path_to_files):
        if len(file_list) > 0:
            if condition_true is not None:
                if not any(c in root for c in condition_true):
                    continue
            if condition_false is not None:
                if any(c in root for c in condition_false):
                    continue
            if 'Raw' in path_to_files:
                if any('Log.csv' in file for file in file_list) or any('TLD_Mirror' in file for file in file_list):
                    name = os.path.split(root)[1]
                    date = regex.search("(\d{6})|(\d*-[\d-]*\d)", root).group(0)
                    material = root.split(rf'\{date}')[0].split('Raw\\')[1]
                    data_files[rf'{date}_{name}']={}
                    data_files[rf'{date}_{name}']['Date'] = date
                    data_files[rf'{date}_{name}']['Material'] = material
                    data_files[rf'{date}_{name}']['Raw_path'] = root
            if 'Processed' in path_to_files:
                if any('stack.tif' in file for file in file_list) and not any(f in root for f in ['Metadata','Colour-out']):
                    name = os.path.split(root)[1]
                    date = regex.search("(\d{6})|(\d*-[\d-]*\d)", root).group(0)
                    material = root.split(rf'\{date}')[0].split('Processed\\')[1]
                    data_files[rf'{date}_{name}']={}
                    data_files[rf'{date}_{name}']['Date'] = date
                    data_files[rf'{date}_{name}']['Material'] = material
                    data_files[rf'{date}_{name}']['Processed_path'] = root
                    data_files[rf'{date}_{name}']['data']={}
                    if load_data:
                        data_files[rf'{date}_{name}']['data'] = data(data_files[rf'{date}_{name}']['Processed_path'])
                    else:
                        data_files[rf'{date}_{name}']['data']['stack_meta'] = data(data_files[rf'{date}_{name}']['Processed_path']).stack_meta
                    if os.path.exists(root.replace('Processed','Raw')):
                        data_files[rf'{date}_{name}']['Raw_path'] = root.replace('Processed','Raw')
                    else:
                        data_files[rf'{date}_{name}']['Raw_path'] = 'not known at this address'
    return data_files                      

def load(folder, factor_helios=-0.39866666666666667, factor_nova=1/2.84, corr=6, AC=True): #add roi input here
    name = os.path.split(folder)[1]
    if 'Processed' in folder and os.path.exists(rf'{folder}\Metadata'):
        processed = True
        stacks = glob.glob(rf'{folder}\*stack*.tif')
        for stack_file in stacks:
            if '_AC' in stack_file:
                stack = tf.imread(stack_file)
            if 'aligned' in stack_file:
                stack = tf.imread(stack_file)
            else:
                stack = tf.imread(stack_file)
        dtype_info = np.iinfo(stack.dtype)
        stack_meta_files = glob.glob(rf'{folder}\Metadata\*stack_meta*.json')
        if len(stack_meta_files)==0 and '_AC' in os.path.split(folder)[1]:
            stack_meta_files = glob.glob(rf'{folder.split("_AC")[0]}\Metadata\*stack_meta*.json')
            with open(stack_meta_files[0]) as file:
                stack_meta = json.load(file)
            file.close()
        else:
            with open(stack_meta_files[0]) as file:
                stack_meta = json.load(file)
            file.close()
        sys, analyser = sys_type(stack_meta['img1'])
        
        ana_voltage = []
        if sys == True:
            for page in stack_meta:
                ana_voltage.append(stack_meta[page]['TLD']['Mirror'])
        if sys == False:
            if stack_meta['img1']['TLD']['Deflector'] in stack_meta['img1']['TLD']:
                for page in stack_meta:
                    ana_voltage.append(stack_meta[page]['TLD']['Deflector'])
            else:
                print('Warning, no Deflector in stack_meta, searching raw data for Log.csv')
                ana_voltage = np.loadtxt(rf"{folder.replace('Raw','Processed')}\Log.csv",delimiter=',', skiprows=2)[:,1]
        if sys==True:
            eV = np.array(ana_voltage)*factor_helios+corr
        if sys==False:
            eV = np.array((ana_voltage*factor_nova))
        
    if 'Raw' in folder:
        processed = False
        files = glob.glob(rf'{folder}\*.tif')
        ana_voltage = []
        for file in files:
            metadata = load_single_file(file, load_img = False)
            if 'Helios' in metadata['System']['SystemType']:
                sys = True
                analyser = 'Mirror'
                ana_voltage.append(metadata['TLD']['Mirror'])
        if 'Nova' in metadata['System']['SystemType']:
            sys = False
            analyser = 'Deflector'
            ana_voltage = np.loadtxt(rf'{folder}\Log.csv',delimiter=',', skiprows=2)[:,1]
            files.sort(key=lambda f: int(''.join(filter(str.isdigit, f) or -1)))
            files_sorted = files
        if sys == True:
            files_sorted = [files for (ana_voltage,files) in sorted(zip(ana_voltage,files), key=lambda pair: pair[0], reverse=sys)]
            ana_voltage = []
        y,x = metadata['Image']['ResolutionY'], metadata['Image']['ResolutionX']
        hfw = metadata['Scan']['HorFieldsize']
        imgs = []
        stack_meta = {}
        ref_img = tf.imread(files_sorted[-1])
        dtype_info = np.iinfo(ref_img.dtype)
        template,temp_path, area = template_crop(ref_img,y,x,hfw)
        shift_list_1 = []
        for i,file in enumerate(files_sorted):
            i+=1 # for legacy reasons (img saved from TLD_Mirror1)
            img, metadata = load_single_file(file)
            stack_meta[f'img{i}'] = metadata
            reg_img, shift_y, shift_x = align_img_template(ref_img,img,template,y,x,temp_path[0,0],temp_path[0,1])
            imgs.append(np.array(reg_img*dtype_info.max,dtype=img.dtype))
            shift_list_1.append([shift_x,shift_y])
            stack_meta[f'img{i}']['Processing'] = {}
            stack_meta[f'img{i}']['Processing']['file'] = file
            stack_meta[f'img{i}']['Processing']['transformation'] = {}
            stack_meta[f'img{i}']['Processing']['transformation']['x'] = float(shift_x)
            stack_meta[f'img{i}']['Processing']['transformation']['y'] = float(shift_y)
            if sys == True:
                ana_voltage.append(metadata['TLD'][analyser])
            if sys == False:
                stack_meta[f'img{i}']['TLD'][analyser] = ana_voltage[i-1]
        x_max, y_max = np.ceil(np.max(shift_list_1,axis=0))
        x_min, y_min = np.floor(np.min(shift_list_1,axis=0))
        if y_min > 0:
            y_min = 0
        if x_min > 0:
            x_min = 0
        stack = np.array(imgs)[:,int(0-y_min):int(y-y_max),int(0-x_min):int(x-x_max)]
        stack_meta[f'img{len(stack)}']['Processing']['temp_match'] = {}
        stack_meta[f'img{len(stack)}']['Processing']['temp_match']['ref_img'] = ref_img[0:y,0:x].tolist()
        stack_meta[f'img{len(stack)}']['Processing']['temp_match']['path'] = temp_path.tolist()
        stack_meta[f'img{len(stack)}']['Processing']['temp_match']['area'] = area
        if sys==True:
            eV = np.array(ana_voltage)*factor_helios+corr
        if sys==False:
            eV = np.array((ana_voltage*factor_nova))
    return stack, stack_meta, eV, dtype_info, name

def load_single_file(file, load_img = True):
    with tf.TiffFile(file) as tif:
        metadata = tif.fei_metadata
    tif.close()
    if load_img == True:
        img = tf.imread(file)
        return img, metadata
    else:
        return metadata

def sys_type(metadata):
    """
    Gives system type of the FEI/ThermoFisher microscope
    and analyser type 'Deflector' for Nova, 'Mirror' for Helios

    Parameters
    ----------
    metadata_page : python dict
        A single page of FEI/ThermoFisher image metadata

    Returns
    -------
    sys, True for Helios, False for Nova
    analyser, type, 'Deflector' or 'Mirror'
    """
    if 'Helios' in metadata['System']['SystemType']:
        sys = True
        analyser = 'Mirror'
    if 'Nova' in metadata['System']['SystemType']:
        sys = False
        analyser = 'Deflector'
    return sys, analyser

def metadata_warning(stack_meta):
    #do something about working distance, suction voltage, dwell, you know
    return

def template_crop(ref_img,y,x,hfw):
    w = hfw*1e6
    area = w*3/1040 + 303/520
    height,width = int(area*y),int(area*x)
    yi,yii = int(y/2-height/2),int(y/2+height/2)
    xi,xii = int(x/2-width/2),int(x/2+width/2)
    temp_path = np.array([[xi,yi],[xii,yi],[xii,yii],[xi,yii],[xi,yi]])
    return ref_img[yi:yii,xi:xii], temp_path, area

def align_img_template(ref_img,mov_img,template,y,x,yi,xi):
    if y == None:
        y = ref_img.shape[0]
    if x == None:
        x = ref_img.shape[1]
    mov_img = mov_img[0:y,0:x]
    result = match_template(np.array(mov_img,dtype='uint8'), np.array(template,dtype='uint8'), TM_CCOEFF_NORMED)
    minV, maxV, minpt, maxpt = min_max_loc(result)
    shift_x,shift_y = np.asarray(maxpt)-(xi,yi)
    tform = transform.EuclideanTransform(translation = (shift_x,shift_y))
    reg_img = transform.warp(mov_img, tform)
    #reg_img = np.array(reg_img)
    return reg_img, shift_y, shift_x

def align_img_pcc(ref_img, mov_img, crop_y=None, crop_x=None, upsample_factor = 10):
    if crop_y == None:
        crop_y = ref_img.shape[0]
    if crop_x == None:
        crop_x = ref_img.shape[1]
    mov_img = mov_img[0:crop_y,0:crop_x]
    shift,err,phasediff = pcc(ref_img[0:crop_y,0:crop_x], mov_img, upsample_factor=upsample_factor)
    shift_y,shift_x=shift
    tform = transform.EuclideanTransform(translation = (shift_x,shift_y))
    reg_img = transform.warp(mov_img, tform)
    reg_img = np.array(reg_img)
    return reg_img, shift_y, shift_x

def conversion(stack_meta, factor, corr):
    MV = []
    for page in stack_meta:
        stack_meta[page]['TLD']['Mirror']
    eV = (MV*factor)+corr
    return eV

def plot_axes(norm=False):
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.it'] = 'sans:italic:bold'
    plt.rcParams['mathtext.bf'] = 'sans:bold'
    plt.xlabel('Energy, $\mathit{E}$ [eV]',weight='bold')
    if norm == False:
        plt.ylabel('Emission intensity [arb.u.]',weight='bold')
    if norm == True:
        plt.ylabel('Emission intensity norm. [arb.u.]',weight='bold')

def plot_scalebar(img, length_fraction=0.3, font_size=15, stack_meta=None, metadata=None, pixel_width=None, hfw=None, img_info=None, save_path=None):
    plt.imshow(img, cmap="gray")
    plt.axis('off')
    if stack_meta is not None:
        pixel_width = stack_meta['img1']['Scan']['PixelWidth']
    if metadata is not None:
        pixel_width = metadata['Scan']['PixelWidth']
    #pixel_width case all good
    if hfw is not None:
        pixel_width = hfw/img.shape[1]
    if img_info is not None:
        with tf.TiffFile(img_info) as tif:
            pixel_width = tif.fei_metadata['Scan']['PixelWidth']
        tif.close()
    #else:
    #    print('Provide pixel width info')
    #    return
    scalebar = ScaleBar(dx=pixel_width, length_fraction=0.3, location='lower right', border_pad=0.5, color='w', box_color='k', box_alpha=0.5, font_properties={'size':'15'})
    plt.gca().add_artist(scalebar)
    if save_path is not None:
        plt.savefig(save_path, dpi=400, bbox_inches='tight',pad_inches=0)
        plt.show()
    if save_path is None:
        plt.show()
        
def zpro(stack):
    zpro = []
    for i in np.arange(0,stack.shape[0],1):
        zpro.append(np.mean(stack[i,:,:]))
    return zpro

def spec_dose(stack_meta):
    """
    Returns the electron dose per image expressed through an overall charge exposure, 
    given by the product of the beam current I_0 and the total exposure time t_total per area A,
    whereby typical quoted units are coulombs per metre squared [Cm^-2]

    Parameters
    ----------
    stack_meta : python dict
        dict of FEI/ThermoFisher metadata throughout stack

    Returns
    -------
    Float
        Spectrum dose [Cm^-2]
    """
    d_img_list = []
    for page in stack_meta:
        I_0 = stack_meta[page]['EBeam']['BeamCurrent']
        t_dwell = stack_meta[page]['Scan']['Dwelltime']
        n_px = stack_meta[page]['Image']['ResolutionX']*stack_meta[page]['Image']['ResolutionY']
        line_int = stack_meta[page]['EScan']['ScanInterlacing']
        n_average = stack_meta[page]['Scan']['Average']
        A = stack_meta[page]['Scan']['HorFieldsize']*stack_meta[page]['Scan']['VerFieldsize']
        d_img = ((I_0*t_dwell*n_px*n_average)/A)/line_int
        d_img_list.append(d_img)
        d_spec = np.sum(d_img_list)
    if stack_meta[page]['Processing']['angular_correction'] in stack_meta:
        d_spec = 2*d_spec
    return d_spec

def roi_masks(img, rois_data):
    if len(img.shape) == 3:
        img_r = img[-1,:,:]
        z,y,x = img.shape
    if len(img.shape) == 2:
        img_r = img
        y,x = img.shape
    if '.zip' in rois_data:
        rois = load_roi_file(rois_data)
        for name in rois:
            ygrid, xgrid = np.mgrid[:y, :x]
            xypix = np.vstack((xgrid.ravel(), ygrid.ravel())).T
            img_mask = np.ma.getmask(np.ma.array(img_r, mask=False))    # initialise the False 2D mask that roi_paths will be added to
            for key in rois[name]['roi_path']:                          # loop through rois in composite roi
                pth = Path(rois[name]['roi_path'][key], closed=False)       # construct a Path from the vertices
                mask = pth.contains_points(xypix)                           # test which pixels fall within the path
                mask = mask.reshape(y,x)                                    # reshape to the same size as the image
                img_mask = np.ma.mask_or(img_mask,mask)                     # add the xycrop to the 2D mask
            rois[name]['img_mask'] = img_mask                           # add img mask to rois dict
    if type(rois_data) is dict:
        rois=rois_data
        for name in rois:
            ygrid, xgrid = np.mgrid[:y, :x]
            xypix = np.vstack((xgrid.ravel(), ygrid.ravel())).T
            img_mask = np.ma.getmask(np.ma.array(img_r, mask=False))
            pth = Path(rois[name]['roi_path'], closed=False)
            mask = pth.contains_points(xypix)
            mask = mask.reshape(y,x)
            img_mask = np.ma.mask_or(img_mask,mask)
            rois[name]['img_mask'] = img_mask
    if '.npy' in rois_data:
        rois = {}
        masks = np.load(rois_data)
        i=0
        while i <= masks.max():
            rois[i] = {}
            rois[i]['img_mask'] = np.where(masks==i,True,False)
            i+=1
    return rois

def load_roi_file(path_to_roi_file):
    """
    Gets the xy points to draw a roi from the imageJ .roi file to a python dictionary
    the read_roi module can not write imageJ compatible roi files
    Parameters
    ----------
    path_to_roi_file : str
        path to imageJ .roi file or .zip of files

    Returns
    -------
    rois : dict
        dict of rois with properties such as type, xy_crop

    """
    #if '.zip' in path_to_roi_file:
    r=read_roi.read_roi_zip(path_to_roi_file)
    #if '.roi' in path_to_roi_file:
    #    r=read_roi.read_roi_file(path_to_roi_file)
    for name in r:
        xy_crop = {}
        if r[name]['type'] == 'rectangle':
            x1 = r[name]['left']
            x2 = x1 + r[name]['width']
            y1 = r[name]['top']
            y2 = y1 + r[name]['height']
            xc = np.array([x1,x2,x2,x1])
            yc = np.array([y1,y1,y2,y2])
            xy_crop[0] = np.vstack((xc, yc)).T
        if r[name]['type'] == 'composite':
            for i,path in enumerate(r[name]['paths']):
                xc = np.array(r[name]['paths'][i])[:,0]
                yc = np.array(r[name]['paths'][i])[:,1]
                xy_crop[i] = np.vstack((xc, yc)).T
        if r[name]['type'] == 'freehand':
            xc = np.array(r[name]['x'])
            yc = np.array(r[name]['y'])
            xy_crop[0] = np.vstack((xc, yc)).T
        r[name]['roi_path'] = xy_crop
    return r

class data:
    factor = -0.39866666666666667
    corr = 6
    def __init__(self, folder, AC=True):
        if AC is True and 'Raw' in folder and os.path.exists(rf'{folder}_R'):
            self.folder = rf'{folder}_AC'
            stack_r_file = rf'{folder}_R'
            stack,stack_meta,self.eV,self.dtype_info,name = load(folder)
            stack_r, stack_meta_r, eV_r, dtype_info_r, name_r = load(stack_r_file)
            for page,page_r in zip(stack_meta,stack_meta_r):
                stack_meta[page]['Processing']['angular_correction'] = 'True'
                stack_meta[page]['Processing']['transformation_r'] = stack_meta_r[page]['Processing']['transformation']
                stack_meta[page]['Processing']['file_r'] = stack_meta_r[page]['Processing']['file']
            stack = img_as_float(stack)
            stack_r = img_as_float(stack_r)
            temp1=stack[-1,:,:]
            temp2=stack_r[-1,:,:]
            temp2 = transform.rotate(temp2,180)
            stack_r = scnd.rotate(stack_r,180,axes=(2,1))
            y1,x1 = temp1.shape
            y2,x2 = temp2.shape
            if y1 == y2:
                yicrop = 'Equal'
                py = 0
                yi = temp1.shape[0]
            if y1 < y2:
                py = y2-y1
                temp1=np.insert(temp1,(y2-y1)*[0],0,axis=0)
                stack=np.insert(stack,(y2-y1)*[0],0,axis=1)
                yi = temp1.shape[0]
                yicrop=True
            if y1 > y2:
                py = y1-y2
                temp2=np.insert(temp2,(y1-y2)*[0],0,axis=0)
                stack_r=np.insert(stack_r,(y1-y2)*[0],0,axis=1)
                yi = temp2.shape[0]
                yicrop=False
            if x1 == x2:
                xicrop = 'Equal'
                px = 0
                xi = temp1.shape[1]
            if x1 < x2:
                px = x2-x1
                temp1=np.insert(temp1,(x2-x1)*[0],0,axis=1)
                stack=np.insert(stack,(x2-x1)*[0],0,axis=2)
                xi = temp1.shape[1]
                xicrop=True
            if x1 > x2:
                px = x1-x2
                temp2=np.insert(temp2,(x1-x2)*[0],0,axis=1)
                stack_r=np.insert(stack_r,(x1-x2)*[0],0,axis=2)
                xi = temp2.shape[1]
                xicrop=False
            shifts, err, phasediff = pcc(temp1, temp2) #transformation required for reg
            ty,tx = shifts
            stack_r = scnd.shift(stack_r,shift=[0,ty,tx])
            halfDiff = (stack-stack_r)/2 #what about average stack?
            stackCorr = stack-halfDiff
            stackCorr = stackCorr*self.dtype_info.max
            stack_AC = stackCorr.astype(self.dtype_info.dtype)
            if yicrop == 'Equal':
                if ty>0:
                    stack_AC_crop = stack_AC[:,int(py+ty):int(yi+ty),:]
                if ty<0 or ty==0:
                    y0=0
                    if abs(ty)<py:
                        y0 = int(py)
                    stack_AC_crop = stack_AC[:,y0:int(yi+ty),:]
            if yicrop == False:
                if ty>0:
                    stack_AC_crop = stack_AC[:,int(py+ty):int(yi+ty),:]
                if ty<0 or ty==0:
                    y0=0
                    if abs(ty)<py:
                        y0 = int(py)
                    stack_AC_crop = stack_AC[:,y0:int(yi+ty),:]
            if yicrop == True:
                if ty>py:
                    y0 = int(ty)
                else:
                    y0 = py
                stack_AC_crop = stack_AC[:,y0:int(yi+ty),:]
            if xicrop == 'Equal':
                if tx>0:
                    stack_AC_crop = stack_AC[:,:,:]
                if tx<0 or tx==0:
                    x0=0
                    if abs(tx)<px:
                        x0 = int(px)
                    stack_AC_crop = stack_AC[:,:,:]
            if xicrop == False:
                if tx>0:
                    stack_AC_crop = stack_AC_crop[:,:,int(tx+px):xi]
                if tx<0 or tx==0:
                    x0=0
                    if abs(tx)<px:
                        x0 = int(px)
                    stack_AC_crop = stack_AC_crop[:,:,x0:int(xi+tx)]
            if xicrop == True:
                if tx>px:
                    x0 = int(tx)
                if tx>0:
                    stack_AC_crop = stack_AC_crop[:,:,x0:xi]
                if tx<0 or tx==0:
                    stack_AC_crop = stack_AC_crop[:,:,x0:int(xi+tx)]
            self.stack, self.stack_meta = stack_AC_crop, stack_meta
            self.stack_meta_r = stack_meta_r
            self.name = rf'{name}_AC'
            self.shape = self.stack.shape
        else:
            self.folder = folder
            self.stack, self.stack_meta, self.eV, self.dtype_info, self.name = load(folder)
            self.shape = self.stack.shape
    def spec(self, rois = None):
        if rois is not None:
            if type(rois) is not dict:
                r = roi_masks(self.stack, rois)
            if 'img_mask' not in rois[list(rois.keys())[0]]:
                r = roi_masks(self.stack, rois)
            else:
                r = rois
            for name in r:
                stack_mask = np.array([r[name]['img_mask']]*self.shape[0])
                stack_masked = np.ma.masked_array(self.stack, ~stack_mask)
                r[name]['spec'] = np.gradient(zpro(stack_masked))
            return r
        else:
            spec = np.gradient(zpro(self.stack))
            return spec
    def spec_dose(self):
        return spec_dose(self.stack_meta)
    def plot_template_roi(self):
        ref_img = self.stack_meta[f'img{len(self.stack)}']['Processing']['temp_match']['ref_img']
        temp_path = self.stack_meta[f'img{len(self.stack)}']['Processing']['temp_match']['path']
        area = self.stack_meta[f'img{len(self.stack)}']['Processing']['temp_match']['area']
        plt.imshow(ref_img)
        plt.plot(temp_path[:,0],temp_path[:,1],c='r')
        plt.text(temp_path[0,0]+10,temp_path[0,1]-15,rf'template, area = {round(area,3)}',backgroundcolor=(1,1,1,0.3))
        plt.show()
    def plot_reg_tforms(self):
        for i,page in enumerate(self.stack_meta):
            plt.plot(i,self.stack_meta[page]['Processing']['transformation']['x'], 'o', color='r')
            plt.plot(i,self.stack_meta[page]['Processing']['transformation']['y'], 'o', color='b')
        plt.legend(['x_shift','y_shift'])
        if '_AC' in self.name and self.stack_meta['img1']['Processing']['angular_correction'] in self.stack_meta['img1']:
            for i,page in enumerate(self.stack_meta):
                plt.plot(i,self.stack_meta[page]['Processing']['transformation_r']['x'], 'ks', markerfacecolor='none', color='c')
                plt.plot(i,self.stack_meta[page]['Processing']['transformation_r']['y'], 'ks', markerfacecolor='none', color='m')
            plt.legend(['x_shift','y_shift','x_shift_r','y_shift_r'])
        plt.xlabel('Slice no.')
        plt.ylabel('Translation [pixel]')
        plt.show()
    def img_avg(self):
        img = np.array(np.mean(self.stack,axis=0),dtype=self.dtype_info.dtype)
        return img
    def plot_img(self, scalebar=True, plot=True, fin_img=False):
        if fin_img == False:
            img = data.img_avg(self)
        if fin_img == True:
            img = self.stack[-1,:,:]
        if scalebar == True:
            plot_scalebar(img, stack_meta=self.stack_meta)
        if scalebar == False:
            plt.imshow(img, cmap='gray')
            plt.axis('off')
        if plot is True:
            plt.show()
    def plot_spec(self, rois=None, plot=True):
        if rois is None:
            plt.plot(self.eV, data.spec(self))
            plot_axes()
            if plot:
                plt.show()
        if rois is not None:
            if type(rois) is not dict:
                r = roi_masks(self.stack, rois)
            if 'img_mask' not in rois[list(rois.keys())[0]]:
                r = roi_masks(self.stack, rois)
            else:
                r = rois
            color = cm.rainbow(np.linspace(0,1,len(r)))
            masks=np.empty(self.shape[1:3])
            for i, (name,c) in enumerate(zip(r,color)):
                if not 'spec' in r[name]:
                    r[name]['spec'] = data.spec(self, rois)[name]['spec']
                plt.plot(self.eV, data.spec(self, rois)[name]['spec'], c=c, label=name)
                img_mask = np.where(r[name]['img_mask']==True,i+1,0)
                masks = masks+img_mask
            masks = masks-1
            norm = colors.Normalize(vmin=0, vmax=len(r))
            cmap = plt.get_cmap('rainbow')
            masks_n = norm(masks)
            rgba = cmap(masks_n)
            rgba[masks==-1,:] = [1,1,1,1]
            plot_axes()
            plt.legend()
            if plot:
                plt.show()
                #n=len(r)+1
                plt.imshow(rgba)
                
    def plot_zpro(self):
        plt.plot(self.eV, data.zpro(self))
        plot_axes()
        plt.show()
    def save_data(self, save_path = None):
        if save_path is None:
            if 'Raw' in self.folder:
                save_path = self.folder.replace('Raw','Processed')
            if 'Processed' in self.folder:
                save_path = self.folder
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)
        pixel_width_um = self.stack_meta['img1']['Scan']['PixelWidth']*1e6
        tf.imwrite(rf"{save_path}\{self.name}_avg_img.tif",
                   data=data.img_avg(self), dtype=self.dtype_info.dtype, photometric='minisblack', imagej=True, 
                   resolution=(1./pixel_width_um, 1./pixel_width_um), metadata={'unit': 'um', 'axes':'YX'})
        labels = []
        for page in self.stack_meta:
            labels.append(os.path.split(self.stack_meta[page]['Processing']['file'])[1])
        tf.imwrite(rf'{save_path}\{self.name}_stack.tif',
                   self.stack, dtype=self.dtype_info.dtype, photometric='minisblack', imagej=True,
                   resolution=(1./pixel_width_um, 1./pixel_width_um), metadata={'spacing':1, 'unit': 'um', 'axes':'ZYX', 'Labels':labels}) #make numpy array into multi page OME-TIF format (Bio - formats)
        
        if os.path.exists(rf'{save_path}\Metadata') is False:
            os.makedirs(rf'{save_path}\Metadata')
        with open(rf'{save_path}\Metadata\{self.name}_stack_meta.json', 'w') as f:
            json.dump(self.stack_meta, f)
        f.close()
        if '_AC' in self.name:
            with open(rf'{save_path}\Metadata\{self.name}_stack_meta_r.json', 'w') as f:
                json.dump(self.stack_meta_r, f)
            f.close()
        plot_scalebar(data.img_avg(self), stack_meta=self.stack_meta, save_path=rf'{save_path}\{self.name}_avg_img_scaled.tif')