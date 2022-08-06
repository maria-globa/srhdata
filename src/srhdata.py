#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 16:50:21 2022

@author: maria
"""
from srhFits.srhFits import SrhFitsFile

def open(filename):
    srhFitsObj = SrhFitsFile(filename)
    return srhFitsObj