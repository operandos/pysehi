# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 16:30:30 2022

@author: James Nohl
"""

import xlsxwriter
import pysehi as ps
import os
import numpy as np

def summary_excel(path_to_files, date:int=None, condition_true:list=None, condition_false:list=None,):
    
    data = ps.list_files(path_to_files, date, condition_true, condition_false, load_data=True)
    
    ### Initialise workbook ###
    init_path = os.path.split(data[list(data.keys())[0]]['Processed_path'])[0]
    init_mat = data[list(data.keys())[0]]['Material']
    init_date = data[list(data.keys())[0]]['Date']
    #workbook = xlsxwriter.Workbook(rf"{init_path}\{init_date}_{init_mat}_specOut.xlsx")
    workbook = xlsxwriter.Workbook(rf"{init_path}\specOut.xlsx")
    worksheetSpec = workbook.add_worksheet('FOV_spec')
    #worksheetMeta = workbook.add_worksheet('Metadata')
    
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
        #worksheetSpec.insert_image(row_num+3, 0+colNoSpec, rf'{DES}\{specNo}\{specNo}_AVG-scaled.png', {'x_scale':0.555, 'y_scale':0.555})
        """
        #do metedata output
        worksheetMeta.write(1,0+colNoMeta,name)
        worksheetMeta.write(2,0+colNoMeta,stack_meta['img1']['User']['Date'])
        worksheetMeta.write(2,1+colNoMeta,'Start')
        worksheetMeta.write(2,3+colNoMeta,'End')
        worksheetMeta.write(2,2+colNoMeta,f'{min(Times)}')
        worksheetMeta.write(2,4+colNoMeta,f'{max(Times)}')
        worksheetMeta.write(3,0+colNoMeta, 'HFW [um]')
        worksheetMeta.write(3,2+colNoMeta, fileMetadata['EScan']['HorFieldsize']*1e6)
        worksheetMeta.write(4,0+colNoMeta,'Beam voltage [V]')
        worksheetMeta.write(4,2+colNoMeta,fileMetadata['Beam']['HV'])
        #Use beam current for Helios
        if switch == 'Nova':
            worksheetMeta.write(5,0+colNoMeta,'Emission current / A')
            worksheetMeta.write(5,2+colNoMeta,f'{emissionAverage} +- {float(emissionDifference)/2}')
            worksheetMeta.write(5,3+colNoMeta,f'+-{round_sig(((float(emissionDifference)/2)/float(emissionAverage))*100,3)}%')
        else:
            worksheetMeta.write(5,0+colNoMeta,'Beam current / A')
            worksheetMeta.write(5,2+colNoMeta,f'{beamAverage} +- {beamDifference/2}')
            if beamDifference == 0:
                worksheetMeta.write(5,3+colNoMeta,f'+-{beamDifference}%')
            else:
                worksheetMeta.write(5,3+colNoMeta,f'+-{round_sig(((beamDifference/2)/beamAverage)*100,3)}%')
        #
        worksheetMeta.write(6,0+colNoMeta, 'Chamber Pressure / Pa')
        worksheetMeta.write(6,2+colNoMeta,f'{ChPressureAverage} +- {ChPressureDifference/2}')
        worksheetMeta.write(7,0+colNoMeta, 'Stage: R,T,X,Y')
        worksheetMeta.write(7,1+colNoMeta, fileMetadata['Stage']['StageR'])
        worksheetMeta.write(7,2+colNoMeta, fileMetadata['Stage']['StageT'])
        worksheetMeta.write(7,3+colNoMeta, fileMetadata['Stage']['StageX'])
        worksheetMeta.write(7,4+colNoMeta, fileMetadata['Stage']['StageY'])
        worksheetMeta.write(8,0+colNoMeta, 'Working distance / mm')
        worksheetMeta.write(8,2+colNoMeta, fileMetadata['Stage']['WorkingDistance']*1e3)
        worksheetMeta.write(9,0+colNoMeta, 'D_spec / Cm^-2')
        worksheetMeta.write(9,2+colNoMeta,D_spec)
        if switch == 'Nova':
            worksheetMeta.write(10,0+colNoMeta, 'defStep')
            worksheetMeta.write(10,1+colNoMeta, deflectorStep)
            worksheetMeta.write(10,2+colNoMeta, 'defFin')
            worksheetMeta.write(10,3+colNoMeta, deflectorFinal)
        else:
            worksheetMeta.write(10,0+colNoMeta, 'MV step / V')
            worksheetMeta.write(10,2+colNoMeta, MVstep)
            worksheetMeta.write(11,0+colNoMeta, 'Suction tube / V')
            worksheetMeta.write(11,2+colNoMeta, fileMetadata['TLD']['SuctionTube'])
        ###
        worksheetMeta.write(12,0+colNoMeta, 'Time')
        for row_num, data in enumerate(Times):            # Timestamp
            worksheetMeta.write(row_num+13, 0+colNoMeta, data)
        worksheetMeta.write(12, 2+colNoMeta, 'xShift')
        for row_num, data in enumerate(shiftArray[:,0]):   # xShift
            worksheetMeta.write(row_num+13, 2+colNoMeta, data)
        worksheetMeta.write(12, 3+colNoMeta, 'yShift')
        for row_num, data in enumerate(shiftArray[:,1]):   # yShift
            worksheetMeta.write(row_num+13, 3+colNoMeta, data)
        if switch == 'Nova':
            worksheetMeta.write(12, 1+colNoMeta, 'Def V / V')
            for row_num, data in enumerate(defV):  # Nova EmissionCurrents
                worksheetMeta.write(row_num+13, 1+colNoMeta, data)
            worksheetMeta.write(12, 4+colNoMeta, 'Emis. cur.')
            for row_num, data in enumerate(EmissionCurrents):  # Nova EmissionCurrents
                worksheetMeta.write(row_num+13, 4+colNoMeta, data)
        else:
            worksheetMeta.write(12, 1+colNoMeta, 'MV')
            for row_num, data in enumerate(MVs):  # Helios BeamCurrents
                worksheetMeta.write(row_num+13, 1+colNoMeta, data)
            worksheetMeta.write(12, 4+colNoMeta, 'Beam cur.')
            for row_num, data in enumerate(BeamCurrents):  # Helios BeamCurrents
                worksheetMeta.write(row_num+13, 4+colNoMeta, data)
        worksheetMeta.write(12, 5+colNoMeta, 'ChPressure')
        for row_num, data in enumerate(ChPressures):       # ChPressures
            worksheetMeta.write(row_num+13, 5+colNoMeta, data)
        worksheetMeta.insert_image(row_num+15,0+colNoMeta, DES+"\\"+subDir+"\\Metadata\\"+specNo+"_metaPlots.png", {'x_scale':0.6149, 'y_scale':0.6105})
        """
        countNo+=1
    worksheetSpec.insert_chart(0,4+colNoSpec, chartNorm)
    worksheetSpec.insert_chart(15,4+colNoSpec, chartIntensity)
    #worksheetMeta.write(0,0,SEM)
    workbook.close()
    
def experimental_conditions(stack_meta):
    #so something to compare acquisitions and warn of variation