# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:12:09 2023

@author: James Nohl
"""

from deepdiff import DeepDiff
from engineering_notation import EngNumber

def metadata_params(stack_meta=None, parameter=None, readable=True):
    params=[]
    if type(parameter) is str:
        params.append(parameter)
    if type(parameter) is list:
        params=parameter
    if stack_meta is None:
        keys={}
    prop_list = ['curr','accel','uc','wd','r','x','y','z','hfw','cntr','brtn','average','interlacing','dwell', 'step', 'range']
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
            if prop == "brtn":
                k,unit = ['TLD','Brightness'], '%'
            if prop == "cntr":
                k,unit = ['TLD','Contrast'], '%'
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
                if readable:
                    if len(unit)>0:
                        print('{:<15s} {:<15s} {:<15s}'.format(prop,rf'{EngNumber(value)}',unit))
                    if len(unit)==0:
                        print('{:<15s} {:<15s}'.format(prop,value))
                if readable is False:
                    if parameter is not None:
                        return k, value
    if stack_meta is None:
        return keys

def compare_params(stack_meta_1, stack_meta_2, readable=True, condition_true:list=None):
    diff_out = DeepDiff(stack_meta_1['img1'],stack_meta_2['img1'], exclude_paths=["root['PrivateFEI']","root['PrivateFei']","root['User']", "root['System']", "root['Processing']", "root['GIS']"])
    diff={}
    for page in diff_out['values_changed']:
        diff[page.split("'")[1]]={}
    for page in diff_out['values_changed']:
        diff[page.split("'")[1]][page.split("'")[3]]=diff_out['values_changed'][page]
        keys = metadata_params(parameter=condition_true)
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