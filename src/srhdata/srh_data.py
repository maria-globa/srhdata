#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 13:57:36 2022

@author: mariagloba
"""

from .srh_fits_0306 import SrhFitsFile0306
from .srh_fits_0612 import SrhFitsFile0612
from .srh_fits_1224 import SrhFitsFile1224
from astropy.io import fits

def same_array(files):
    files = list(files)
    if len(files) > 1:
        f = fits.open(files[0])
        srh_array = f[0].header['INSTRUME']
        for file in files[1:]:
            f = fits.open(file)
            if srh_array != f[0].header['INSTRUME']:
                return False
    return True

def open(files):
    if type(files) == str:
        files = [files,]
    if type(files) == list:
        pass
    else:
        raise TypeError('Input value must be string or list')
    if same_array(files):
        f = fits.open(files[0])
        srh_array = f[0].header['INSTRUME']
        if srh_array == 'SRH0306':
            srh_obj = SrhFitsFile0306(files)
        elif srh_array == 'SRH0612':
            srh_obj = SrhFitsFile0612(files)
        elif srh_array == 'SRH1224':
            srh_obj = SrhFitsFile1224(files)
        else:
            raise UnknownArray(srh_array)
    else:
        raise Exception("Input data must be measured by the same array (3-6, 6-12 or 12-24)")
    return srh_obj

def execute_task(synth_task):
    files = []
    calibrate = True
    for i in range(len(synth_task['task'])):
        files.append(synth_task['task'][i]['file_path'])
    if same_array(files):
        f = fits.open(files[0])
        srh_array = f[0].header['INSTRUME']
        if srh_array == 'SRH0306':
            srh_obj = SrhFitsFile0306(files)
        elif srh_array == 'SRH0612':
            srh_obj = SrhFitsFile0612(files)
        elif srh_array == 'SRH1224':
            srh_obj = SrhFitsFile1224(files)
        else:
            raise UnknownArray(srh_array)
    else:
        raise Exception("Input data must be measured by the same array (3-6, 6-12 or 12-24)")
    srh_obj.select_scans(synth_task)
    out_pol = synth_task['params']['output_polarizations'] == 'RL'
    if 'gains' in synth_task.keys():
        srh_obj.loadGains(synth_task['gains'])
        calibrate = False
    srh_obj.makeImage(path = synth_task['params']['outdir'],
                      frequency = synth_task['task'][0]['frequency_index'],
                      average = srh_obj.dataLength,
                      #compress_image = synth_task['params']['compressed'],
                      RL = out_pol,
                      clean_disk = synth_task['params']['clean_disk'],
                      cdelt = synth_task['params']['cdelt'],
                      naxis = synth_task['params']['naxis'], 
                      calibrate = calibrate)
    if out_pol:
        return {'R' : srh_obj.out_filenames[0], 'L' : srh_obj.out_filenames[1]}
    else:
        return {'I' : srh_obj.out_filenames[0], 'V' : srh_obj.out_filenames[1]}

def execute_task_calibration(synth_task):
    files = []
    for i in range(len(synth_task['task'])):
        files.append(synth_task['task'][i]['file_path'])
    if same_array(files):
        f = fits.open(files[0])
        srh_array = f[0].header['INSTRUME']
        if srh_array == 'SRH0306':
            srh_obj = SrhFitsFile0306(files)
        elif srh_array == 'SRH0612':
            srh_obj = SrhFitsFile0612(files)
        elif srh_array == 'SRH1224':
            srh_obj = SrhFitsFile1224(files)
        else:
            raise UnknownArray(srh_array)
    else:
        raise Exception("Input data must be measured by the same array (3-6, 6-12 or 12-24)")
    srh_obj.select_scans(synth_task)
    srh_obj.calibrate(freq = synth_task['task'][0]['frequency_index'])
    return srh_obj.getGains(synth_task['task'][0]['frequency_index'])
  

class UnknownArray(Exception):
    def __init__(self, srh_array, message="SRH array is unknown"):
        self.message = "SRH array \"%s\" is unknown" % srh_array
        super().__init__(self.message)