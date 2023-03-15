# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:12:09 2023

@author: James Nohl
"""

from deepdiff import DeepDiff
from engineering_notation import EngNumber
import csv

def metadata_params(data=None, parameter=None, parameter_false=None,readable=True, write=False):
    stack_meta = data.stack_meta
    name=rf"{data.date}_{data.name}"
    params=[]
    if type(parameter) is str:
        params.append(parameter)
    if type(parameter) is list:
        params=parameter
    params_false=[]
    if type(parameter_false) is str:
        params_false.append(parameter_false)
    if 'stage' in params_false:
        params_false.extend(['r','t','x','y','z'])
        params_false.remove('stage')
    if type(parameter_false) is list:
        params_false=parameter_false
    prop_list = ['curr','accel','uc','dwell','wd','r','t','x','y','z','hfw','px','py','average','integrate','interlacing','cntr','brtn']#,'step', 'range']
    if parameter is None:
        params=prop_list
    if type(params_false) is list:
        for prop_f in params_false:
            if any(prop_f in p for p in params):
                params.remove(prop_f)
    if write is True:
        with open(rf"{data.folder.replace('Raw','Processed')}\Metadata\{data.name}_stack_meta_readable.txt", "a") as f:
            f.write(f'{name.center(42)}\n')
            f.write('_'*42)
            f.write('\n')
            f.write('{:<15s} {:<15s} {:<15s}'.format('parameter','value','unit'))
            f.write('\n')
            f.write('_'*42)
            f.write('\n')
    if readable is True:
        print('\n',rf'{name.center(42)}')
        print('_'*42)
        print('{:<15s} {:<15s} {:<15s}'.format('parameter','value','unit'))
        print('_'*42)
    if write is True:
        rows=[]
    if stack_meta is None:
        keys={}
    i=1
    for prop in params:
        if any(prop in p for p in prop_list):
            i+=1
            if prop == 'curr':
                k,unit = ['EBeam','BeamCurrent'], 'A'
            if prop == 'accel':
                k,unit = ['Beam','HV'], 'V'
            if prop == 'uc':
                k,unit = ['EBeam','BeamMode'], ''
            if prop == "dwell":
                k,unit = ['EScan','Dwell'], 's'
            if prop == "wd":
                k,unit = ['Stage','WorkingDistance'], 'm'
            if prop == "hfw":
                k,unit = ['EScan','HorFieldsize'], 'm'
            if prop == "px":
                k,unit = ['Image', 'ResolutionX'], 'pixels'
            if prop == "py":
                k,unit = ['Image', 'ResolutionY'], 'pixels'
            if prop == "average":
                k,unit = ['Image','Average'], 'frames'
            if prop == "integrate":
                k,unit = ['Image','Integrate'], 'frames'
            if prop == "interlacing":
                k,unit = ['EScan','ScanInterlacing'], 'lines'
            if prop == "step" or prop == "range":
                k,unit = ['TLD','Mirror'], 'V'
            if prop == "r":
                k,unit = ['Stage','StageR'], 'rad'
            if prop == "t":
                k,unit = ['Stage','StageT'], 'rad'
            if prop == "brtn":
                k,unit = ['TLD','Brightness'], '%'
            if prop == "cntr":
                k,unit = ['TLD','Contrast'], '%'
            if any(prop in s for s in ['x','y','z']):
                k,unit = ['Stage',rf'Stage{prop.capitalize()}'], 'm'
            if stack_meta is None:
                keys[prop]=k
            if type(stack_meta) is dict:
                if 'img1' in stack_meta:
                    value = stack_meta['img1']
                if 'img1' not in stack_meta:
                    value = stack_meta
                for key in k:
                    value = value.get(key)
                    if not value:
                        break
                if type(value) is float or type(value) is int: 
                    v=str(EngNumber(value))
                else:
                    v=value
                if write:
                    with open(rf"{data.folder.replace('Raw','Processed')}\Metadata\{data.name}_stack_meta_readable.txt", "a") as f:
                        f.write(str('{:<15s} {:<15s} {:<15s}'.format(key,v,unit)))
                        f.write('\n')
                if readable:
                    print('{:<15s} {:<15s} {:<15s}'.format(prop,v,unit))
                if readable is False:
                    if parameter is not None:
                        return k, value
    if write:
        f.close()
    if stack_meta is None:
        return keys

def compare_params(meta_1, meta_2, readable=True, condition_true:list=None):
    if 'img1' in meta_1.keys():
        meta_1 = meta_1['img1']
    if 'img1' in meta_2.keys():
        meta_2 = meta_2['img1']
    diff_out = DeepDiff(meta_1,meta_2, exclude_paths=["root['PrivateFEI']","root['PrivateFei']","root['User']", "root['System']", "root['Processing']", "root['GIS']"])
    diff={}
    for page in diff_out['values_changed']:
        diff[page.split("'")[1]]={}
    for page in diff_out['values_changed']:
        diff[page.split("'")[1]][page.split("'")[3]]=diff_out['values_changed'][page]
        keys = metadata_params(parameter=condition_true,readable=False)
        """
        val = diff
        for keyc in keys:
            for kc in keys[keyc]:
                val=val.get(kc)
                if not val:
                    break
                print(val)
        """
    return diff
    
def wd_check(stack_meta, readable=True, distance=0.0040):
    """
    check that the working distance is 4.0 mm
    
    Parameters
    ----------
    stack_meta : dict
        dict of stack metadata.
    readable : bool, optional
        If readable is True, print result. The default is True.

    Returns
    -------
    bool
        True - the wd is 4 mm.

    """
    if round(metadata_params(stack_meta, 'wd', readable=False)[1],4) == distance:
        if readable is False:
            return True
    if round(metadata_params(stack_meta, 'wd', readable=False)[1],4) != distance:
        if readable is True:
            print('\t\tWARNING !','\t\t', 'Working distance is \t', metadata_params(stack_meta, 'wd', readable=False)[1]*1000,' mm')
        if readable is False:
            return False