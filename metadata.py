# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:12:09 2023

@author: James Nohl
"""

from deepdiff import DeepDiff
from engineering_notation import EngNumber

def metadata_params(stack_meta=None, parameter=None, check_prop=False, readable=True):
    params=[]
    if parameter is not None:
        params.append(parameter)
    prop_list = ['curr','accel','uc','wd','r','x','y','z','hfw','average','interlacing','dwell', 'step', 'range']
    if parameter is None:
        params=prop_list
    for prop in params:
        if any(prop in p for p in prop_list):
            if prop == 'curr':
                k,unit = ['EBeam','BeamCurrent'], 'A'
            if prop == 'accel':
                k,unit = ['Beam','HV'], 'V'
            if prop == 'uc':
                k,unit = ['EBeam','BeamMode'], ''
            if prop == "wd":
                k,unit = ['Stage','WorkingDistance'], 'm'
            if prop == "hfw":
                k,unit = ['EScan','HorFieldsize'], 'm'
            if prop == "interlacing":
                k,unit = ['EScan','ScanInterlacing'], 'lines'
            if prop == "dwell":
                k,unit = ['EScan','Dwell'], 's'
            if prop == "average":
                k,unit = ['Image','Average'], 'frames'
            if prop == "step" or prop == "range":
                k,unit = ['TLD','Mirror'], 'V'
            if any(prop in s for s in ['r','x','y','z']):
                k,unit = ['Stage',rf'Stage{prop.capitalize()}'], 'm'
            if stack_meta is None:
                return k
            if type(stack_meta) is dict:
                value = stack_meta['img1']
                for key in k:
                    value = value.get(key)
                    if not value:
                        break
                if readable:
                    if len(unit)>0:
                        print('{:<15s} {:<15s} {:<15s}'.format(prop,rf'{EngNumber(value)}',unit))
                    if len(unit)==0:
                        print('{:<15s} {:<15s}'.format(prop,value))
                if readable is False:
                    return k, value

def compare_params(stack_meta_1, stack_meta_2, drop:list=None):
    diff = DeepDiff(stack_meta_1['img1'],stack_meta_2['img1'], exclude_paths=["root['PrivateFEI']","root['PrivateFei']","root['User']", "root['System']", "root['Processing']", "root['GIS']"])
    return diff
    
def params_warning(stack_meta):
    if round(metadata_params(stack_meta, 'wd', readable=False)[1],4) != 0.0040:
        print('\t\tWARNING !','\t\t', 'Working distance is \t', metadata_params(stack_meta, 'wd', readable=False)[1]*1000,' mm')