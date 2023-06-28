#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 10:07:15 2022

@author: mariagloba
"""
import numpy as NP

def real_to_complex(z):
    return z[:len(z)//2] + 1j * z[len(z)//2:]
    
def complex_to_real(z):
    return NP.concatenate((NP.real(z), NP.imag(z)))

def wrap(value):
    while value<-180:
        value+=360
    while value>180:
        value-=360
    return value

def createDisk(size, radius = 980, arcsecPerPixel = 2.45552):
    qSun = NP.zeros((size, size))
    sunRadius = radius / (arcsecPerPixel)
    for i in range(size):
        x = i - size//2 - 1
        for j in range(size):
            y = j - size//2 - 1
            if (NP.sqrt(x*x + y*y) < sunRadius):
                qSun[i , j] = 1
    return qSun

def ihhmm_format(t):
    hh = int(t / 3600.)
    t -= hh*3600.
    mm = int(t / 60.)
    t -= mm*60.
    ss = int(t)
    return '%02d:%02d:%02d' % (hh,mm,ss)