#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 16:50:21 2022

@author: maria
"""
<<<<<<< HEAD
from srhFits import SrhFitsFile
=======
from srhFits.srhFits import SrhFitsFile
>>>>>>> 5bc4c92d5a6af9ca52690b9017a0b0e4a794ec67

def open(filename):
    srhFitsObj = SrhFitsFile(filename)
    return srhFitsObj