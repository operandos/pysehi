# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 16:30:30 2022

@author: James Nohl
"""

import xlsxwriter
import pysehi as ps
import metadata as md
import os
import numpy as np
import regex
import tifffile as tf
from skimage import transform
import scipy.ndimage as scnd
from cv2 import matchTemplate as match_template
from cv2 import TM_CCOEFF_NORMED
from cv2 import minMaxLoc as min_max_loc
import math
import matplotlib.pyplot as plt
import pathlib

def summary_excel(path_to_files, date:int=None, condition_true:list=None, condition_false:list=None, custom_name=None):
    
    data = ps.list_files(path_to_files, date, condition_true, condition_false, load_data=True, custom_name=None)
    
    paths=[]
    for key in list(data.keys()):
        paths.append(os.path.split(data[key]['Processed_path'])[0])
    folder_groups = list(set(paths))
    for fg in folder_groups:
        slash = slash_type(fg)
        ### Initialise workbook ###
        init_path = pathlib.Path(fg)
        for part in init_path.parts:
            init_date = regex.search("(\d{6})|(\d*-[\d-]*\d)", part)
            if init_date is not None:
                init_date=init_date.group(0)
                break
        if fg.find(init_date)>fg.find('Processed'):
            if type(pathlib.Path(fg)) is pathlib.WindowsPath:
                init_mat = fg.split(rf'\{init_date}')[0].split('Processed\\')[1].replace('\\','_')
            else:
                init_mat = fg.split(rf'/{init_date}')[0].split('Processed/')[1].replace('/','_')
        if fg.find(init_date)<fg.find('Processed'):
            if type(pathlib.Path(fg)) is pathlib.WindowsPath:
                init_mat = fg.split('Reference data\\')[1].split('\\')[0]
            else:
                init_mat = fg.split('Reference data/')[1].split('/')[0]
        if type(pathlib.Path(fg)) is pathlib.WindowsPath:
            if len(fg.split(f'{init_date}\\')) > 1:
                init_exp = fg.split(f'{init_date}\\')[1].replace('\\','_')
                workbook = xlsxwriter.Workbook(rf"{init_path}\{init_date}_{init_mat}_{init_exp}_specOut.xlsx")
            else:
                workbook = xlsxwriter.Workbook(rf"{init_path}\{init_date}_{init_mat}_specOut.xlsx")
        else:
            if len(fg.split(f'{init_date}/')) > 1:
                init_exp = fg.split(f'{init_date}/')[1].replace('/','_')
                workbook = xlsxwriter.Workbook(rf"{init_path}/{init_date}_{init_mat}_{init_exp}_specOut.xlsx")
            else:
                workbook = xlsxwriter.Workbook(rf"{init_path}/{init_date}_{init_mat}_specOut.xlsx")
        worksheetSpec = workbook.add_worksheet('FOV_spec')
        #worksheetMeta = workbook.add_worksheet('Metadata')
        
        ### FOV spec plots ###
        chartIntensity = workbook.add_chart({'type':'scatter','subtype':'straight'})
        chartIntensity.set_title ({'name':rf'{init_date}_{init_mat}', 'name_font':{'size':12}})
        chartIntensity.set_x_axis({'name':'Energy [eV]'})
        chartIntensity.set_y_axis({'name':'Emission intensity [arb.u.]', 'major_gridlines':{'visible': False}})
        chartIntensity.set_legend({'position':'top'})
        
        chartNorm = workbook.add_chart({'type':'scatter','subtype':'straight'})
        chartNorm.set_title ({'name':rf'{init_date}_{init_mat}_norm', 'name_font':{'size':12}})
        chartNorm.set_x_axis({'name':'Energy [eV]'})
        chartNorm.set_y_axis({'name':'Emission intensity norm [arb.u.]','major_gridlines': {'visible': False}})
        chartNorm.set_legend({'position':'top'})
        
        ### add data to workbook ###
        countNo = 0
        for name in data:
            if fg not in data[name]['Processed_path']:
                continue
            stack_meta = data[name]['data'].stack_meta
            eV = data[name]['data'].eV
            zpro = ps.zpro(data[name]['data'].stack)
            spec = data[name]['data'].spec()
            #img_avg = data[name]['data'].img_avg()
            colNoSpec = countNo*4
            #colNoMeta = countNo*7
            worksheetSpec.write(0,0+colNoSpec,name)
            worksheetSpec.write(0,1+colNoSpec, f"HFW = {stack_meta['img1']['EScan']['HorFieldsize']*1e6} um")
            worksheetSpec.write(1,0+colNoSpec,'Energy [eV]')
            worksheetSpec.write(1,1+colNoSpec,'Z-profile')
            worksheetSpec.write(1,2+colNoSpec,'Emission intensity')
            worksheetSpec.write(1,3+colNoSpec,'Emission intensity norm')
            for row_num, value in enumerate(eV):    # Energy / eV
                worksheetSpec.write(row_num+2, 0+colNoSpec, value)
            for row_num, value in enumerate(zpro):          # Z-profile
                worksheetSpec.write(row_num+2, 1+colNoSpec, value)
            for row_num, value in enumerate(spec):     # spec
                worksheetSpec.write(row_num+2, 2+colNoSpec, value)
            for row_num, value in enumerate(spec/np.max(spec)): # spec norm
                worksheetSpec.write(row_num+2, 3+colNoSpec, value)
            #set up the chart series
            chartNorm.add_series({
                'name':         name,
                'categories':   ['FOV_spec', 2, 0+colNoSpec, 203, 0+colNoSpec],
                'values':       ['FOV_spec', 2, 3+colNoSpec, 203, 3+colNoSpec]
            })
            chartIntensity.add_series({
                'name':         name,
                'categories':   ['FOV_spec', 2, 0+colNoSpec, 203, 0+colNoSpec],
                'values':       ['FOV_spec', 2, 2+colNoSpec, 203, 2+colNoSpec]
            })
            f_name = name.split(rf"{init_date}_")[1]
            if os.path.exists(rf"{data[name]['Processed_path']}{slash}{f_name}_avg_img_scaled.png"):
                worksheetSpec.insert_image(row_num+3, 0+colNoSpec, rf"{data[name]['Processed_path']}{slash}{f_name}_avg_img_scaled.png", {'x_scale':0.555, 'y_scale':0.555})
            if os.path.exists(rf"{data[name]['Processed_path']}{slash}Metadata{slash}{f_name}_stack_meta_plots.png"):
                worksheetSpec.insert_image(row_num+12, 0+colNoSpec, rf"{data[name]['Processed_path']}{slash}Metadata{slash}{f_name}_stack_meta_plots.png", {'x_scale':0.48, 'y_scale':0.48})
            countNo+=1
        worksheetSpec.insert_chart(0,4+colNoSpec, chartNorm)
        worksheetSpec.insert_chart(15,4+colNoSpec, chartIntensity)
        workbook.close()

def location_mosaic(path_to_folder, path_to_img_overview=None, condition_false=None,path_to_img_template=None):
    slash=slash_type(path_to_folder)
    loc_dict = {}
    if path_to_img_overview is not None:
        if '.tif' in path_to_img_overview:
            img_over, meta_over = ps.load_single_file(path_to_img_overview)
        else:
            dat_over = ps.data(path_to_img_overview)
            img_over, meta_over = dat_over.img_avg(), dat_over.stack_meta['img1']
    
        ### load img_over_data
        loc_dict['loc_img'] = {}
        loc_dict['loc_img']['r'] = meta_over['Stage']['StageR']
        loc_dict['loc_img']['t'] = meta_over['Stage']['StageT']
        loc_dict['loc_img']['x'] = meta_over['Stage']['StageX']
        loc_dict['loc_img']['y'] = meta_over['Stage']['StageY']
        loc_dict['loc_img']['z'] = meta_over['Stage']['StageZ']
        loc_dict['loc_img']['Hor'] = meta_over['Scan']['HorFieldsize']
        loc_dict['loc_img']['Ver'] = meta_over['Scan']['VerFieldsize']
        
        ### get corners of img_over
        l,r = loc_dict['loc_img']['x']-loc_dict['loc_img']['Hor']/2,loc_dict['loc_img']['x']+loc_dict['loc_img']['Hor']/2
        b,t = loc_dict['loc_img']['y']-loc_dict['loc_img']['Ver']/2,loc_dict['loc_img']['y']+loc_dict['loc_img']['Ver']/2
        img_over_corners = np.array([[l,r,r,l],[t,t,b,b]]).T
        
        ### prepare the img_over by cropping to image region and rotating ###
        ResX,ResY = meta_over['Image']['ResolutionX'],meta_over['Image']['ResolutionY']
        rot_anticlock = -(loc_dict['loc_img']['r']/np.pi)*180
        img_over_r = transform.rotate(img_over[0:ResY,0:ResX], rot_anticlock, center=[0,0], resize=True)
    
    ### load mosaic regions metadata into dict ###
    files = ps.list_files(path_to_folder,condition_false=condition_false, load_data=True)
    
    ### add mosaic region properties to loc_dict ###
    paths=np.empty([1,2])
    for sub in files:
        loc_dict[sub] = {}
        stack_metadata = files[sub]['data'].stack_meta
        
        for prop in ['r','t','x','y','z']:
            k, val = md.metadata_params(files[sub]['data'], prop, readable=False)
            loc_dict[sub][prop] = val
        loc_dict[sub]['Hor'] = stack_metadata['img1']['Scan']['HorFieldsize']
        loc_dict[sub]['Ver'] = stack_metadata['img1']['Scan']['VerFieldsize']

        
        x1,x2 = loc_dict[sub]['x']-loc_dict[sub]['Hor']/2,loc_dict[sub]['x']+loc_dict[sub]['Hor']/2
        y1,y2 = loc_dict[sub]['y']+loc_dict[sub]['Ver']/2,loc_dict[sub]['y']-loc_dict[sub]['Ver']/2
        
        path = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2],[x1,y1]])
        p_rs = np.empty([1,2])
        ### rotate points in paths around the origin ###
        for p in path:
            p_r = np.array(rotate([0,0], p, loc_dict[sub]['r']))
            p_rs = np.vstack([p_rs,p_r])
        loc_dict[sub]['path'] = np.vstack(p_rs)
        loc_dict[sub]['path'] = np.delete(loc_dict[sub]['path'],0,0)
        
        paths = np.vstack([paths,path])
    
    ### get the min&max x and y of the mosaic paths to find max extent of the plots ###
    paths = np.delete(paths,0,0)
    x_min,x_max = np.min(paths[:,0]),np.max(paths[:,0])
    y_min,y_max = np.min(paths[:,1]),np.max(paths[:,1])
    
    lic_r = np.empty([1,2])
    for p in img_over_corners:
        p_r = np.array(rotate([0,0], p, loc_dict['loc_img']['r']))
        lic_r = np.vstack([lic_r,p_r])
    lic_r = np.delete(lic_r,0,0)
    
    e_l, e_r = np.min(lic_r[:,0]), np.max(lic_r[:,0])
    e_b, e_t = np.min(lic_r[:,1]), np.max(lic_r[:,1])
    
    plt.imshow(img_over_r,cmap='gray',extent=[e_l*1e6,e_r*1e6,e_b*1e6,e_t*1e6])
    for sub in loc_dict:
        if 'loc_img' not in sub:
            label = sub.split('_',maxsplit=1)[1]
            plt.plot(loc_dict[sub]['path'][:,0]*1e6,loc_dict[sub]['path'][:,1]*1e6,label=label)
            plt.text(loc_dict[sub]['path'][:,0][0]*1e6,loc_dict[sub]['path'][:,1][0]*1e6,label, c=[189/255,195/255,199/255,1])
    plt.xlabel('x [\u03BCm]')
    plt.ylabel('y [\u03BCm]')
    plt.savefig(rf'{path_to_folder}{slash}locations_overview.png',dpi=400,transparent='True')
    plt.show()
    
    ### handle case where the mosaic regions are aligned by template matching ###
    if path_to_img_template is not None:
        if '.tif' in path_to_img_template:
            img_temp, meta_temp = ps.load_single_file(path_to_img_template)
        else:
            dat_temp = ps.data(path_to_img_template)
            img_temp, meta_temp = dat_temp.img_avg(), dat_temp.stack_meta['img1']
        
        ### add template_img properties to the mosaic locations dictionary ###
        loc_dict['template_img'] = {}
        for prop in ['r','t','x','y','z']:
            k, val = md.metadata_params(meta_temp, prop, readable=False)
            loc_dict['template_img'][prop] = val
        loc_dict['template_img']['Hor'] = meta_temp['Scan']['HorFieldsize']
        loc_dict['template_img']['Ver'] = meta_temp['Scan']['VerFieldsize']
        
        tx1,tx2 = loc_dict['template_img']['x']-loc_dict['template_img']['Hor']/2,loc_dict['template_img']['x']+loc_dict['template_img']['Hor']/2
        ty1,ty2 = loc_dict['template_img']['y']+loc_dict['template_img']['Ver']/2,loc_dict['template_img']['y']-loc_dict['template_img']['Ver']/2
        t_path = np.array([[tx1,ty1],[tx2,ty1],[tx2,ty2],[tx1,ty2],[tx1,ty1]])
        p_rs = np.empty([1,2])
        ### rotate points in paths around the origin ###
        for p in t_path:
            p_r = np.array(rotate([0,0], p, loc_dict['template_img']['r']))
            p_rs = np.vstack([p_rs,p_r])
        loc_dict['template_img']['path'] = np.vstack(p_rs)
        loc_dict['template_img']['path'] = np.delete(loc_dict['template_img']['path'],0,0)
        
        ### prepare the template image rotation ###
        temp_ResX,temp_ResY = meta_temp['Image']['ResolutionX'],meta_temp['Image']['ResolutionY']
        temp_rot_anticlock = -(loc_dict['template_img']['r']/np.pi)*180
        img_temp_r = transform.rotate(img_temp[0:temp_ResY,0:temp_ResX], temp_rot_anticlock, center=[0,0], resize=True)
        
        ### resample img_temp_r to img_over pixelwidth ###
        img_over_pwidth = meta_over['Scan']['PixelWidth']
        img_temp_pwidth = meta_temp['Scan']['PixelWidth']
        scale_factor = img_temp_pwidth/img_over_pwidth
        img_temp_rrs = transform.resize(img_temp_r, output_shape = (int(img_temp_r.shape[0]*scale_factor),int(img_temp_r.shape[1]*scale_factor)))
        
        ### pad img_temp to img_over size for pcc alignment ###
        ypad = img_over_r.shape[0] - img_temp_rrs.shape[0]
        xpad = img_over_r.shape[1] - img_temp_rrs.shape[1]
        img_temp_rrsp = np.pad(img_temp_rrs, ((0,ypad),(0,xpad)),mode='constant')
        
        ### align with pcc and translate
        result = match_template(np.array(img_temp_rrs,dtype='uint8'), np.array(img_over_r,dtype='uint8'), TM_CCOEFF_NORMED)
        minV, maxV, minpt, maxpt = min_max_loc(result)
        tx,ty = np.asarray(maxpt)
        img_temp_rrsps = scnd.shift(img_temp_rrsp,shift=[ty,tx])
            
def rotate(origin, point, angle):
    """
    Rotate a point clockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(-angle) * (px - ox) - math.sin(-angle) * (py - oy)
    qy = oy + math.sin(-angle) * (px - ox) + math.cos(-angle) * (py - oy)
    return qx, qy
#def experimental_conditions(stack_meta):
    #so something to compare acquisitions and warn of variation
    
def slash_type(path):
    if type(pathlib.Path(path)) is pathlib.WindowsPath:
        slash = '\\'
    else:
        slash = '/'
    return slash