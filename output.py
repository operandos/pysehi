# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 16:30:30 2022

@author: James Nohl
"""

import xlsxwriter
import pysehi as ps
import os
import numpy as np
import regex

def summary_excel(path_to_files, date:int=None, condition_true:list=None, condition_false:list=None, custom_name=None):
    
    data = ps.list_files(path_to_files, date, condition_true, condition_false, load_data=True, custom_name=None)
    
    paths=[]
    for key in list(data.keys()):
        paths.append(os.path.split(data[key]['Processed_path'])[0])
    folder_groups = list(set(paths))
    for fg in folder_groups:
        ### Initialise workbook ###
        init_path = fg
        init_date = regex.search("(\d{6})|(\d*-[\d-]*\d)", fg).group(0)
        init_mat = fg.split(rf'\{init_date}')[0].split('Processed\\')[1].replace('\\','_')
        if len(fg.split(f'{init_date}\\')) > 1:
            init_exp = fg.split(f'{init_date}\\')[1].replace('\\','_')
            workbook = xlsxwriter.Workbook(rf"{init_path}\{init_date}_{init_mat}_{init_exp}_specOut.xlsx")
        else:
            workbook = xlsxwriter.Workbook(rf"{init_path}\{init_date}_{init_mat}_specOut.xlsx")
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
            if os.path.exists(rf"{data[name]['Processed_path']}\{f_name}_avg_img_scaled.png"):
                worksheetSpec.insert_image(row_num+3, 0+colNoSpec, rf"{data[name]['Processed_path']}\{f_name}_avg_img_scaled.png", {'x_scale':0.555, 'y_scale':0.555})
            if os.path.exists(rf"{data[name]['Processed_path']}\Metadata\{f_name}_stack_meta_plots.png"):
                worksheetSpec.insert_image(row_num+12, 0+colNoSpec, rf"{data[name]['Processed_path']}\Metadata\{f_name}_stack_meta_plots.png", {'x_scale':0.48, 'y_scale':0.48})
            countNo+=1
        worksheetSpec.insert_chart(0,4+colNoSpec, chartNorm)
        worksheetSpec.insert_chart(15,4+colNoSpec, chartIntensity)
        workbook.close()
    
#def experimental_conditions(stack_meta):
    #so something to compare acquisitions and warn of variation