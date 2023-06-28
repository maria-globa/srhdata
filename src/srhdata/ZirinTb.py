#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  1 04:20:33 2021

@author: svlesovoi
"""

import numpy as NP
from scipy import constants
#import pylab as PL
import scipy.optimize as opt

def fitFunc(f, A, B, C):
    return A + B*f + C*f**-1.8

#guess  = [1, 1, 1]
#
#frequency = [1.4, 1.6, 1.8, 2.0, 2.4, 2.8,  3.2, 3.6,  4.2, 5.0,  5.8, 7.0, 8.2, 9.4, 10.6, 11.8, 13.2, 14.8, 16.4, 18.0]
#Tb = [70.5, 63.8, 52.2, 42.9, 32.8, 27.1, 24.2, 21.7, 19.4, 17.6, 15.9, 14.1, 12.9, 12.2, 11.3, 11.0, 10.8, 10.8, 10.7, 10.3]
#err = [3.0, 2.8, 2.5, 1.9, 1.4, 1.1, 1.1, 1.1, 0.8, 0.8, 0.7, 0.6, 0.6, 0.6, 0.5, 0.5, 0.5, 0.6, 0.7, 0.5]
#
#flux = 2*constants.k*NP.array(Tb)*1e3/(constants.c/(NP.array(frequency)*1e9))**2 * NP.deg2rad(36/60)**2 / 1e-22
#
#fitTbParams, _ = opt.curve_fit(fitFunc, frequency, Tb, p0=guess)
#
#fittedTb = fitFunc(NP.array(frequency), fitTbParams[0], fitTbParams[1], fitTbParams[2])

#PL.clf()
#PL.plot(frequency, Tb, '.')
#
#srhFrequency = NP.linspace(2,25,100)
#PL.plot(frequency, fittedTb)
#PL.plot(srhFrequency, fitFunc(srhFrequency, fitTbParams[0], fitTbParams[1], fitTbParams[2]))

class ZirinTb():
    def __init__(self):
        self.frequency = NP.array([1.4, 1.6, 1.8, 2.0, 2.4, 2.8,  3.2, 3.6,  4.2, 5.0,  5.8, 7.0, 8.2, 9.4, 10.6, 11.8, 13.2, 14.8, 16.4, 18.0])# frequency [GHz]
        self.Tb = NP.array([70.5, 63.8, 52.2, 42.9, 32.8, 27.1, 24.2, 21.7, 19.4, 17.6, 15.9, 14.1, 12.9, 12.2, 11.3, 11.0, 10.8, 10.8, 10.7, 10.3])# brightness temperature [1e3K]
        self.guess = [1,1,1]
        self.fitTbParams, _ = opt.curve_fit(fitFunc, self.frequency, self.Tb, p0=self.guess)
        self.solarDiskRadius = NP.deg2rad(900/3600)
        
    def getTbAtFrequency(self, f):
        return fitFunc(f, self.fitTbParams[0],self.fitTbParams[1],self.fitTbParams[2])

    def getSfuAtFrequency(self, f):
#        return 2*constants.k*self.getTbAtFrequency(f)*1e3/(constants.c/(f*1e9))**2 * NP.pi*self.solarDiskDiameter**2 / (4*NP.log(2)) / 1e-22
        return 2*constants.k*self.getTbAtFrequency(f)*1e3/(constants.c/(f*1e9))**2 * NP.pi*self.solarDiskRadius**2 / 1e-22

        