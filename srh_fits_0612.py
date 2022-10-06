#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 18:14:20 2022

@author: mariagloba
"""
from .srh_fits import SrhFitsFile
from .srh_coordinates import base2uvw0612
from .srh_uvfits import SrhUVData
from astropy import constants
import numpy as NP
from scipy.optimize import least_squares
import scipy.signal
from . import srh_utils
from skimage.transform import warp, AffineTransform


class SrhFitsFile0612(SrhFitsFile):
    def __init__(self, name):
        super().__init__(name)
        self.base = 4.9
        self.sizeOfUv = 1024
        self.antNumberEW = 128
        self.antNumberNS = 64
        self.centering_ftol = 1e-14
        self.convolutionNormCoef = 18.7
        super().open()
        
    def solarPhase(self, freq):
        u,v,w = base2uvw0612(self.RAO.hAngle, self.RAO.declination, 129, 130)
        baseWave = NP.sqrt(u**2+v**2)*self.freqList[freq]*1e3/constants.c.to_value()
        if baseWave > 120:
            self.nsSolarPhase[freq] = NP.pi
        else:
            self.nsSolarPhase[freq] = 0
        u,v,w = base2uvw0612(self.RAO.hAngle, self.RAO.declination, 1, 2)
        baseWave = NP.sqrt(u**2+v**2)*self.freqList[freq]*1e3/constants.c.to_value()
        if baseWave > 120:
            self.ewSolarPhase[freq] = NP.pi
        else:
            self.ewSolarPhase[freq] = 0       
        
    def calculatePhaseLcp_nonlinear(self, freqChannel):
        redIndexesNS = []
        for baseline in range(1, self.baselines+1):
            for i in range(self.antNumberNS - baseline):
                redIndexesNS.append(NP.where((self.antennaA==128+i) & (self.antennaB==128+i+baseline))[0][0])
        redIndexesEW = []
        for baseline in range(1, self.baselines+1):
            for i in range(self.antNumberEW - baseline):
                redIndexesEW.append(NP.where((self.antennaA==i) & (self.antennaB==i+baseline))[0][0])
             
        if self.averageCalib:
            redundantVisNS = NP.mean(self.visLcp[freqChannel, :20, redIndexesNS], axis = 1)
            redundantVisEW = NP.mean(self.visLcp[freqChannel, :20, redIndexesEW], axis = 1)
            redundantVisAll = NP.append(redundantVisEW, redundantVisNS)
        else:
            redundantVisNS = self.visLcp[freqChannel, self.calibIndex, redIndexesNS]
            redundantVisEW = self.visLcp[freqChannel, self.calibIndex, redIndexesEW]
            redundantVisAll = NP.append(redundantVisEW, redundantVisNS)
            
        for i in range(len(redIndexesNS)):    
            if NP.any(self.flags_ns == self.antennaA[redIndexesNS[i]]) or NP.any(self.flags_ns == self.antennaB[redIndexesNS[i]]):
                redundantVisNS[i]=0.
        for i in range(len(redIndexesEW)):    
            if NP.any(self.flags_ew == self.antennaA[redIndexesEW[i]]) or NP.any(self.flags_ew == self.antennaB[redIndexesEW[i]]):
                redundantVisEW[i]=0.

        ls_res = least_squares(self.allGainsFunc_constrained, self.x_ini_lcp[freqChannel], args = (redundantVisAll, freqChannel), max_nfev = 400)
        self.calibrationResultLcp[freqChannel] = ls_res['x']
        self.x_ini_lcp[freqChannel] = ls_res['x']
        gains = srh_utils.real_to_complex(ls_res['x'][1:])[(self.baselines-1)*2:]
        self.ew_gains_lcp = gains[:self.antNumberEW]
        self.ewAntPhaLcp[freqChannel] = NP.angle(self.ew_gains_lcp)
        self.ns_gains_lcp = gains[self.antNumberEW:]
        self.nsAntPhaLcp[freqChannel] = NP.angle(self.ns_gains_lcp)
        
        norm = NP.mean(NP.abs(gains[NP.abs(gains)>NP.median(NP.abs(gains))*0.6]))
        self.ewAntAmpLcp[freqChannel] = NP.abs(self.ew_gains_lcp)/norm
        self.ewAntAmpLcp[freqChannel][self.ewAntAmpLcp[freqChannel]<NP.median(self.ewAntAmpLcp[freqChannel])*0.5] = 1e6
        self.nsAntAmpLcp[freqChannel] = NP.abs(self.ns_gains_lcp)/norm
        self.nsAntAmpLcp[freqChannel][self.nsAntAmpLcp[freqChannel]<NP.median(self.nsAntAmpLcp[freqChannel])*0.5] = 1e6
        
    def calculatePhaseRcp_nonlinear(self, freqChannel):
        redIndexesNS = []
        for baseline in range(1, self.baselines+1):
            for i in range(self.antNumberNS - baseline):
                redIndexesNS.append(NP.where((self.antennaA==128+i) & (self.antennaB==128+i+baseline))[0][0])
        redIndexesEW = []
        for baseline in range(1, self.baselines+1):
            for i in range(self.antNumberEW - baseline):
                redIndexesEW.append(NP.where((self.antennaA==i) & (self.antennaB==i+baseline))[0][0])
             
        if self.averageCalib:
            redundantVisNS = NP.mean(self.visRcp[freqChannel, :20, redIndexesNS], axis = 1)
            redundantVisEW = NP.mean(self.visRcp[freqChannel, :20, redIndexesEW], axis = 1)
            redundantVisAll = NP.append(redundantVisEW, redundantVisNS)
        else:
            redundantVisNS = self.visRcp[freqChannel, self.calibIndex, redIndexesNS]
            redundantVisEW = self.visRcp[freqChannel, self.calibIndex, redIndexesEW]
            redundantVisAll = NP.append(redundantVisEW, redundantVisNS)
            
        for i in range(len(redIndexesNS)):    
            if NP.any(self.flags_ns == self.antennaA[redIndexesNS[i]]) or NP.any(self.flags_ns == self.antennaB[redIndexesNS[i]]):
                redundantVisNS[i]=0.
        for i in range(len(redIndexesEW)):    
            if NP.any(self.flags_ew == self.antennaA[redIndexesEW[i]]) or NP.any(self.flags_ew == self.antennaB[redIndexesEW[i]]):
                redundantVisEW[i]=0.

        ls_res = least_squares(self.allGainsFunc_constrained, self.x_ini_rcp[freqChannel], args = (redundantVisAll, freqChannel), max_nfev = 400)
        self.calibrationResultRcp[freqChannel] = ls_res['x']
        self.x_ini_rcp[freqChannel] = ls_res['x']
        gains = srh_utils.real_to_complex(ls_res['x'][1:])[(self.baselines-1)*2:]
        self.ew_gains_rcp = gains[:self.antNumberEW]
        self.ewAntPhaRcp[freqChannel] = NP.angle(self.ew_gains_rcp)
        self.ns_gains_rcp = gains[self.antNumberEW:]
        self.nsAntPhaRcp[freqChannel] = NP.angle(self.ns_gains_rcp)
        
        norm = NP.mean(NP.abs(gains[NP.abs(gains)>NP.median(NP.abs(gains))*0.6]))
        self.ewAntAmpRcp[freqChannel] = NP.abs(self.ew_gains_rcp)/norm
        self.ewAntAmpRcp[freqChannel][self.ewAntAmpRcp[freqChannel]<NP.median(self.ewAntAmpRcp[freqChannel])*0.5] = 1e6
        self.nsAntAmpRcp[freqChannel] = NP.abs(self.ns_gains_rcp)/norm
        self.nsAntAmpRcp[freqChannel][self.nsAntAmpRcp[freqChannel]<NP.median(self.nsAntAmpRcp[freqChannel])*0.5] = 1e6
    
    def allGainsFunc_constrained(self, x, obsVis, freq):
        res = NP.zeros_like(obsVis, dtype = complex)
        ewSolarAmp = 1
        nsSolarAmp = NP.abs(x[0])
        x_complex = srh_utils.real_to_complex(x[1:])
    
        ewGainsNumber = self.antNumberEW
        nsSolVisNumber = self.baselines - 1
        ewSolVisNumber = self.baselines - 1
        ewSolVis = NP.append((ewSolarAmp * NP.exp(1j*self.ewSolarPhase[freq])), x_complex[: ewSolVisNumber])
        nsSolVis = NP.append((nsSolarAmp * NP.exp(1j*self.nsSolarPhase[freq])), x_complex[ewSolVisNumber : ewSolVisNumber+nsSolVisNumber])
        ewGains = x_complex[ewSolVisNumber+nsSolVisNumber : ewSolVisNumber+nsSolVisNumber+ewGainsNumber]
        nsGains = x_complex[ewSolVisNumber+nsSolVisNumber+ewGainsNumber :]
        
        solVisArrayNS = NP.array(())
        antAGainsNS = NP.array(())
        antBGainsNS = NP.array(())
        solVisArrayEW = NP.array(())
        antAGainsEW = NP.array(())
        antBGainsEW = NP.array(())
        for baseline in range(1, self.baselines+1):
            solVisArrayNS = NP.append(solVisArrayNS, NP.full(self.antNumberNS-baseline, nsSolVis[baseline-1]))
            antAGainsNS = NP.append(antAGainsNS, nsGains[:self.antNumberNS-baseline])
            antBGainsNS = NP.append(antBGainsNS, nsGains[baseline:])
            
            solVisArrayEW = NP.append(solVisArrayEW, NP.full(self.antNumberEW-baseline, ewSolVis[baseline-1]))
            antAGainsEW = NP.append(antAGainsEW, ewGains[:self.antNumberEW-baseline])
            antBGainsEW = NP.append(antBGainsEW, ewGains[baseline:])
            
        res = NP.append(solVisArrayEW, solVisArrayNS) * NP.append(antAGainsEW, antAGainsNS) * NP.conj(NP.append(antBGainsEW, antBGainsNS)) - obsVis
        return srh_utils.complex_to_real(res)  
    
    def buildEWPhase(self):
        newLcpPhaseCorrection = NP.zeros(self.antNumberEW)
        newRcpPhaseCorrection = NP.zeros(self.antNumberEW)
        for j in range(self.antNumberEW):
                newLcpPhaseCorrection[j] += NP.deg2rad(self.ewSlopeLcp[self.frequencyChannel] * (j - 63.5)) 
                newRcpPhaseCorrection[j] += NP.deg2rad(self.ewSlopeRcp[self.frequencyChannel] * (j - 63.5))
        self.ewLcpPhaseCorrection[self.frequencyChannel, :] = newLcpPhaseCorrection[:]
        self.ewRcpPhaseCorrection[self.frequencyChannel, :] = newRcpPhaseCorrection[:]
        
    def buildNSPhase(self):
        newLcpPhaseCorrection = NP.zeros(self.antNumberNS)
        newRcpPhaseCorrection = NP.zeros(self.antNumberNS)
        for j in range(self.antNumberNS):
                newLcpPhaseCorrection[j] += (NP.deg2rad(-self.nsSlopeLcp[self.frequencyChannel] * (j + 0.5)) + NP.deg2rad(self.nsLcpStair[self.frequencyChannel]))
                newRcpPhaseCorrection[j] += (NP.deg2rad(-self.nsSlopeRcp[self.frequencyChannel] * (j + 0.5)) + NP.deg2rad(self.nsRcpStair[self.frequencyChannel]))
        self.nsLcpPhaseCorrection[self.frequencyChannel, :] = newLcpPhaseCorrection[:]
        self.nsRcpPhaseCorrection[self.frequencyChannel, :] = newRcpPhaseCorrection[:]
        
    def vis2uv(self, scan, phaseCorrect = True, amplitudeCorrect = True, PSF=False, average = 0):
        self.uvLcp = NP.zeros((self.sizeOfUv, self.sizeOfUv), dtype = complex)
        self.uvRcp = NP.zeros((self.sizeOfUv, self.sizeOfUv), dtype = complex)
        flags_ew = NP.where(self.ewAntAmpLcp[self.frequencyChannel]==1e6)[0]
        flags_ns = NP.where(self.nsAntAmpLcp[self.frequencyChannel]==1e6)[0]

        O = self.sizeOfUv//2 + 1
        if average:
            firstScan = scan
            if  self.visLcp.shape[1] < (scan + average):
                lastScan = self.dataLength
            else:
                lastScan = scan + average
            for i in range(self.antNumberNS):
                for j in range(self.antNumberEW):
                    if not (NP.any(flags_ew == j) or NP.any(flags_ns == i)):
                        self.uvLcp[O + i*2, O + (j-64)*2] = NP.mean(self.visLcp[self.frequencyChannel, firstScan:lastScan, i*128+j])
                        self.uvRcp[O + i*2, O + (j-64)*2] = NP.mean(self.visRcp[self.frequencyChannel, firstScan:lastScan, i*128+j])
                        if (phaseCorrect):
                            ewPh = self.ewAntPhaLcp[self.frequencyChannel, j]+self.ewLcpPhaseCorrection[self.frequencyChannel, j]
                            nsPh = self.nsAntPhaLcp[self.frequencyChannel, i]+self.nsLcpPhaseCorrection[self.frequencyChannel, i]
                            self.uvLcp[O + i*2, O + (j-64)*2] *= NP.exp(1j * (-ewPh + nsPh))
                            ewPh = self.ewAntPhaRcp[self.frequencyChannel, j]+self.ewRcpPhaseCorrection[self.frequencyChannel, j]
                            nsPh = self.nsAntPhaRcp[self.frequencyChannel, i]+self.nsRcpPhaseCorrection[self.frequencyChannel, i]
                            self.uvRcp[O + i*2, O + (j-64)*2] *= NP.exp(1j * (-ewPh + nsPh))
                        if (amplitudeCorrect):
                            self.uvLcp[O + i*2, O + (j-64)*2] /= (self.ewAntAmpLcp[self.frequencyChannel, j] * self.nsAntAmpLcp[self.frequencyChannel, i])
                            self.uvRcp[O + i*2, O + (j-64)*2] /= (self.ewAntAmpRcp[self.frequencyChannel, j] * self.nsAntAmpRcp[self.frequencyChannel, i])
                        self.uvLcp[O - (i+1)*2, O - (j-63)*2] = NP.conj(self.uvLcp[O + i*2, O + (j-64)*2])
                        self.uvRcp[O - (i+1)*2, O - (j-63)*2] = NP.conj(self.uvRcp[O + i*2, O + (j-64)*2])
                        
                    

                
        else:
            for i in range(self.antNumberNS):
                for j in range(self.antNumberEW):
                    if not (NP.any(flags_ew == j) or NP.any(flags_ns == i)):
                        self.uvLcp[O + i*2, O + (j-64)*2] = self.visLcp[self.frequencyChannel, scan, i*128+j]
                        self.uvRcp[O + i*2, O + (j-64)*2] = self.visRcp[self.frequencyChannel, scan, i*128+j]
                        if (phaseCorrect):
                            ewPh = self.ewAntPhaLcp[self.frequencyChannel, j]+self.ewLcpPhaseCorrection[self.frequencyChannel, j]
                            nsPh = self.nsAntPhaLcp[self.frequencyChannel, i]+self.nsLcpPhaseCorrection[self.frequencyChannel, i]
                            self.uvLcp[O + i*2, O + (j-64)*2] *= NP.exp(1j * (-ewPh + nsPh))
                            ewPh = self.ewAntPhaRcp[self.frequencyChannel, j]+self.ewRcpPhaseCorrection[self.frequencyChannel, j]
                            nsPh = self.nsAntPhaRcp[self.frequencyChannel, i]+self.nsRcpPhaseCorrection[self.frequencyChannel, i]
                            self.uvRcp[O + i*2, O + (j-64)*2] *= NP.exp(1j * (-ewPh + nsPh))
                        if (amplitudeCorrect):
                            self.uvLcp[O + i*2, O + (j-64)*2] /= (self.ewAntAmpLcp[self.frequencyChannel, j] * self.nsAntAmpLcp[self.frequencyChannel, i])
                            self.uvRcp[O + i*2, O + (j-64)*2] /= (self.ewAntAmpRcp[self.frequencyChannel, j] * self.nsAntAmpRcp[self.frequencyChannel, i])
                        self.uvLcp[O - (i+1)*2, O - (j-63)*2] = NP.conj(self.uvLcp[O + i*2, O + (j-64)*2])
                        self.uvRcp[O - (i+1)*2, O - (j-63)*2] = NP.conj(self.uvRcp[O + i*2, O + (j-64)*2])
                    
        if PSF:
            self.uvLcp[NP.abs(self.uvLcp)>1e-8] = 1
            self.uvRcp[NP.abs(self.uvRcp)>1e-8] = 1

        self.uvLcp[NP.abs(self.uvLcp)<1e-6] = 0.
        self.uvRcp[NP.abs(self.uvRcp)<1e-6] = 0.
        self.uvLcp /= NP.count_nonzero(self.uvLcp)
        self.uvRcp /= NP.count_nonzero(self.uvRcp)
 
    def uv2lmImage(self):
        self.lcp = NP.fft.fft2(NP.roll(NP.roll(self.uvLcp,self.sizeOfUv//2,0),self.sizeOfUv//2,1));
        self.lcp = NP.roll(NP.roll(self.lcp,self.sizeOfUv//2,0),self.sizeOfUv//2,1);
        self.rcp = NP.fft.fft2(NP.roll(NP.roll(self.uvRcp,self.sizeOfUv//2,0),self.sizeOfUv//2,1));
        self.rcp = NP.roll(NP.roll(self.rcp,self.sizeOfUv//2,0),self.sizeOfUv//2,1);
        
    def lm2Heliocentric(self):
        scaling = self.RAO.getPQScale(self.sizeOfUv, NP.deg2rad(self.arcsecPerPixel * (self.sizeOfUv - 1)/3600.)*2, self.freqList[self.frequencyChannel]*1e3)
        scale = AffineTransform(scale=(self.sizeOfUv/scaling[0], self.sizeOfUv/scaling[1]))
        shift = AffineTransform(translation=(-self.sizeOfUv/2,-self.sizeOfUv/2))
        rotate = AffineTransform(rotation = self.RAO.pAngle)
        matrix = AffineTransform(matrix = self.RAO.getPQ2HDMatrix())
        back_shift = AffineTransform(translation=(self.sizeOfUv/2,self.sizeOfUv/2))

        O = self.sizeOfUv//2
        Q = self.sizeOfUv//4
        dataResult0 = warp(self.lcp.real,(shift + (scale + back_shift)).inverse)
        self.lcp = warp(dataResult0,(shift + (matrix + back_shift)).inverse)
        dataResult0 = warp(self.rcp.real,(shift + (scale + back_shift)).inverse)
        self.rcp = warp(dataResult0,(shift + (matrix + back_shift)).inverse)
        dataResult0 = 0
        self.lcp = warp(self.lcp,(shift + (rotate + back_shift)).inverse)[O-Q:O+Q,O-Q:O+Q]
        self.rcp = warp(self.rcp,(shift + (rotate + back_shift)).inverse)[O-Q:O+Q,O-Q:O+Q]
        self.lcp = NP.flip(self.lcp, 1)
        self.rcp = NP.flip(self.rcp, 1)
        
    def createDisk(self, radius, arcsecPerPixel = 2.45552):
        qSun = NP.zeros((self.sizeOfUv, self.sizeOfUv))
        sunRadius = radius / (arcsecPerPixel)
        for i in range(self.sizeOfUv):
            x = i - self.sizeOfUv//2 - 1
            for j in range(self.sizeOfUv):
                y = j - self.sizeOfUv//2 - 1
                if (NP.sqrt(x*x + y*y) < sunRadius):
                    qSun[i , j] = 1
                    
        dL = 2*( 6//2) + 1
        arg_x = NP.linspace(-1.,1,dL)
        arg_y = NP.linspace(-1.,1,dL)
        xx, yy = NP.meshgrid(arg_x, arg_y)
        
        scaling = self.RAO.getPQScale(self.sizeOfUv, NP.deg2rad(arcsecPerPixel*(self.sizeOfUv-1)/3600.), self.freqList[self.frequencyChannel]*1e3)
        scale = AffineTransform(scale=(scaling[0]/self.sizeOfUv, scaling[1]/self.sizeOfUv))
        back_shift = AffineTransform(translation=(self.sizeOfUv/2, self.sizeOfUv/2))
        shift = AffineTransform(translation=(-self.sizeOfUv/2, -self.sizeOfUv/2))
        matrix = AffineTransform(matrix = NP.linalg.inv(self.RAO.getPQ2HDMatrix()))
        rotate = AffineTransform(rotation = -self.RAO.pAngle)
        
        gKern =   NP.exp(-0.5*(xx**2 + yy**2))
        qSmoothSun = scipy.signal.fftconvolve(qSun,gKern) / dL**2
        qSmoothSun = qSmoothSun[dL//2:dL//2+self.sizeOfUv,dL//2:dL//2+self.sizeOfUv]
        smoothCoef = qSmoothSun[512, 512]
        qSmoothSun /= smoothCoef
        self.qSun_el_hd = warp(qSmoothSun,(shift + (rotate + back_shift)).inverse)
        
        res = warp(self.qSun_el_hd, (shift + (matrix + back_shift)).inverse)
        self.qSun_lm = warp(res,(shift + (scale + back_shift)).inverse)
        qSun_lm_fft = NP.fft.fft2(NP.roll(NP.roll(self.qSun_lm,self.sizeOfUv//2,0),self.sizeOfUv//2,1));
        qSun_lm_fft = NP.roll(NP.roll(qSun_lm_fft,self.sizeOfUv//2,0),self.sizeOfUv//2,1)# / self.sizeOfUv;
        # qSun_lm_fft = NP.flip(qSun_lm_fft, 0)
#        qSun_lm_uv = qSun_lm_fft * uvPsf
#        qSun_lm_conv = NP.fft.fft2(NP.roll(NP.roll(qSun_lm_uv,self.sizeOfUv//2+1,0),self.sizeOfUv//2+1,1));
#        qSun_lm_conv = NP.roll(NP.roll(qSun_lm_conv,self.sizeOfUv//2-1,0),self.sizeOfUv//2-1,1);
#        qSun_lm_conv = NP.flip(NP.flip(qSun_lm_conv, 1), 0)
        self.lm_hd_relation[self.frequencyChannel] = NP.sum(self.qSun_lm)/NP.sum(self.qSun_el_hd)
        self.fftDisk = qSun_lm_fft #qSun_lm_conv, 
    
    def createUvUniform(self):
        self.uvUniform = NP.zeros((self.sizeOfUv, self.sizeOfUv), dtype = complex)
        flags_ew = NP.where(self.ewAntAmpLcp[self.frequencyChannel]==1e6)[0]
        flags_ns = NP.where(self.nsAntAmpLcp[self.frequencyChannel]==1e6)[0]
        O = self.sizeOfUv//2 + 1
        for i in range(self.antNumberNS):
            for j in range(self.antNumberEW):
                if not (NP.any(flags_ew == j) or NP.any(flags_ns == i)):
                    self.uvUniform[O + i*2, O + (j-64)*2] = 1
                    self.uvUniform[O - (i+1)*2, O - (j-63)*2] = 1
        self.uvUniform /= NP.count_nonzero(self.uvUniform)

                    
    def createUvPsf(self, T, ewSlope, nsSlope, stair = 0):
        self.uvPsf = self.uvUniform.copy()
        O = self.sizeOfUv//2 + 1
        ewSlope = NP.deg2rad(ewSlope)
        nsSlope = NP.deg2rad(nsSlope)
        ewSlopeUv = NP.linspace(-O * ewSlope/2., O * ewSlope/2., self.sizeOfUv)
        nsSlopeUv = NP.linspace(-O * nsSlope/2., O * nsSlope/2., self.sizeOfUv)
        ewGrid,nsGrid = NP.meshgrid(ewSlopeUv, nsSlopeUv)
        slopeGrid = ewGrid + nsGrid
        slopeGrid[self.uvUniform == 0] = 0
        self.uvPsf *= T * NP.exp(1j * slopeGrid)
        self.uvPsf[O:,:] *= NP.exp(-1j * NP.deg2rad(stair))
        self.uvPsf[:O,:] *= NP.exp(1j * NP.deg2rad(stair))

    def diskDiff_stair(self, x, pol):
        self.createUvPsf(x[0], x[1], x[2], x[3])
        uvDisk = self.fftDisk * self.uvPsf
        if pol == 0:
            diff = self.uvLcp - uvDisk
        if pol == 1:
            diff = self.uvRcp - uvDisk
        return srh_utils.complex_to_real(diff[self.uvUniform!=0])
    
    def findDisk_stair(self):
        self.createDisk(980)
        self.createUvUniform()
        self.x_ini = [1,0,0,0]
        ls_res = least_squares(self.diskDiff_stair, self.x_ini, args = (0,), ftol=self.centering_ftol)
        _diskLevelLcp, _ewSlopeLcp, _nsSlopeLcp, _nsLcpStair = ls_res['x']
        ls_res = least_squares(self.diskDiff_stair, self.x_ini, args = (1,), ftol=self.centering_ftol)
        _diskLevelRcp, _ewSlopeRcp, _nsSlopeRcp, _nsRcpStair = ls_res['x']
        
        if _diskLevelLcp<0:
            _nsLcpStair += 180
            _diskLevelLcp *= -1
        if _diskLevelRcp<0:
            _nsRcpStair += 180
            _diskLevelRcp *= -1
        
        self.diskLevelLcp[self.frequencyChannel] = _diskLevelLcp
        self.diskLevelRcp[self.frequencyChannel] = _diskLevelRcp
        
        Tb = self.ZirinQSunTb.getTbAtFrequency(self.freqList[self.frequencyChannel]*1e-6) * 1e3
        
        self.ewAntAmpLcp[self.frequencyChannel][self.ewAntAmpLcp[self.frequencyChannel]!=1e6] *= NP.sqrt(_diskLevelLcp*self.convolutionNormCoef / Tb)
        self.nsAntAmpLcp[self.frequencyChannel][self.nsAntAmpLcp[self.frequencyChannel]!=1e6] *= NP.sqrt(_diskLevelLcp*self.convolutionNormCoef / Tb)
        self.ewAntAmpRcp[self.frequencyChannel][self.ewAntAmpRcp[self.frequencyChannel]!=1e6] *= NP.sqrt(_diskLevelRcp*self.convolutionNormCoef / Tb)
        self.nsAntAmpRcp[self.frequencyChannel][self.nsAntAmpRcp[self.frequencyChannel]!=1e6] *= NP.sqrt(_diskLevelRcp*self.convolutionNormCoef / Tb)
        
        self.ewSlopeLcp[self.frequencyChannel] = srh_utils.wrap(self.ewSlopeLcp[self.frequencyChannel] + _ewSlopeLcp)
        self.nsSlopeLcp[self.frequencyChannel] = srh_utils.wrap(self.nsSlopeLcp[self.frequencyChannel] + _nsSlopeLcp)
        self.ewSlopeRcp[self.frequencyChannel] = srh_utils.wrap(self.ewSlopeRcp[self.frequencyChannel] + _ewSlopeRcp)
        self.nsSlopeRcp[self.frequencyChannel] = srh_utils.wrap(self.nsSlopeRcp[self.frequencyChannel] + _nsSlopeRcp)
        self.nsLcpStair[self.frequencyChannel] = srh_utils.wrap(self.nsLcpStair[self.frequencyChannel] + _nsLcpStair)
        self.nsRcpStair[self.frequencyChannel] = srh_utils.wrap(self.nsRcpStair[self.frequencyChannel] + _nsRcpStair)

    def centerDisk(self):
        self.findDisk_stair()
        self.buildEWPhase()
        self.buildNSPhase()
      
    def modelDiskConv(self):
        currentDiskTb = self.ZirinQSunTb.getTbAtFrequency(self.freqList[self.frequencyChannel]*1e-6)*1e3
        self.createUvPsf(currentDiskTb/self.convolutionNormCoef,0,0,0)
        self.uvDiskConv = self.fftDisk * self.uvPsf# - self.uvLcp
        qSun_lm = NP.fft.fft2(NP.roll(NP.roll(self.uvDiskConv,self.sizeOfUv//2,0),self.sizeOfUv//2,1));
        qSun_lm = NP.roll(NP.roll(qSun_lm,self.sizeOfUv//2,0),self.sizeOfUv//2,1)# / self.sizeOfUv;
        qSun_lm = NP.flip(qSun_lm, 0)
        self.modelDisk = qSun_lm
        
    def saveAsUvFits(self, filename):
        uv_fits = SrhUVData()
        uv_fits.write_uvfits_0612(self, filename)
    
    def clean(self):
        pass
        