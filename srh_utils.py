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