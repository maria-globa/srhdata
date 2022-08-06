# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 00:18:47 2016

@author: Sergey
"""

from astropy.io import fits


class SrhFitsFile():
    def __init__(self, name):
        self.open(name)
  
    def open(self,name):
        try:
            self.hduList = fits.open(name)
            self.isOpen = True
        except FileNotFoundError:
            print('File %s  not found'%name);
 