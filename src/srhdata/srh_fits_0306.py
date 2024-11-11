#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 11:40:04 2022

@author: mariagloba
"""

from .srh_fits import SrhFitsFile
from .srh_coordinates import base2uvw0306
from .srh_uvfits import SrhUVData
from astropy import constants
import numpy as NP
from scipy.optimize import least_squares
import scipy.signal
from . import srh_utils
from skimage.transform import warp, AffineTransform
from casatasks import tclean, rmtables
from casatools import image as IA
import os
from .ZirinTb import ZirinTb
from astropy.io import fits
import skimage.measure
from pathlib import Path
from sunpy.map.header_helper import make_heliographic_header
from sunpy.coordinates import get_earth
from scipy.interpolate import RegularGridInterpolator
from threadpoolctl import threadpool_limits

class SrhFitsFile0306(SrhFitsFile):
    def __init__(self, name):
        super().__init__(name)
        self.base = 9800
        self.sizeOfUv = 1025
        self.antNumberEW = 97
        self.antNumberNS = 31
        self.baselines = 5
        super().open()
        self.antZeroRow = self.hduList[3].data['ant_zero_row'][:97]
        self.lcpShift = NP.ones(self.freqListLength) # 0-frequency component in the spectrum
        self.rcpShift = NP.ones(self.freqListLength)
        self.fluxLcp = NP.zeros((self.freqListLength, self.dataLength))
        self.fluxRcp = NP.zeros((self.freqListLength, self.dataLength))
        self.convolutionNormCoef = 44.8
        self.out_filenames = []
        if self.corr_amp_exist:
            self.normalizeFlux()
        
    def normalizeFlux(self):
        file = Path(__file__).resolve()
        parent = str(file.parent)
        zerosFits = fits.open(parent + '/srh_0306_cp_zeros.fits')
        skyLcp = zerosFits[2].data['skyLcp_c']
        skyRcp = zerosFits[2].data['skyRcp_c']
        fluxZerosLcp = zerosFits[2].data['skyLcp']
        fluxZerosRcp = zerosFits[2].data['skyRcp']

        fluxNormFits = fits.open(parent + '/srh_0306_cp_fluxNorm.fits')
        fluxNormLcp = fluxNormFits[2].data['fluxNormLcp']
        fluxNormRcp = fluxNormFits[2].data['fluxNormRcp']
        
        antA_new = self.antennaA.copy()
        antA_new[self.antennaA<98] += 30
        antA_new[self.antennaA>=98] -= 98
        
        antB_new = self.antennaB.copy()
        antB_new[self.antennaB<98] += 30
        antB_new[self.antennaB>=98] -= 98
        
        for tt in range(self.dataLength):
            self.ampLcp_c[:,tt,:] = self.ampLcp_c[:,tt,:] - skyLcp
            self.ampRcp_c[:,tt,:] = self.ampRcp_c[:,tt,:] - skyRcp
        self.ampLcp_c[self.ampLcp_c <= 1e5] = 1e8
        self.ampRcp_c[self.ampRcp_c <= 1e5] = 1e8
        
        self.antAmp_Lcp = NP.sqrt(NP.abs(self.ampLcp_c[:,:,antA_new]) * NP.abs(self.ampLcp_c[:,:,antB_new]))
        self.antAmp_Rcp = NP.sqrt(NP.abs(self.ampRcp_c[:,:,antA_new]) * NP.abs(self.ampRcp_c[:,:,antB_new]))
        
        self.visLcp = self.visLcp/self.antAmp_Lcp
        self.visRcp = self.visRcp/self.antAmp_Rcp
            
        ampFluxRcp = NP.mean(self.ampRcp, axis = 2)
        ampFluxLcp = NP.mean(self.ampLcp, axis = 2)

            
        self.tempRcp = NP.zeros((self.freqListLength, self.dataLength))
        self.tempLcp = NP.zeros((self.freqListLength, self.dataLength))
        
        self.beam()
        
        ampFluxRcp -= fluxZerosRcp[:, None]
        ampFluxRcp *= fluxNormRcp[:, None] 
        ampFluxLcp -= fluxZerosLcp[:, None]
        ampFluxLcp *= fluxNormLcp[:, None] 

        self.fluxLcp = ampFluxLcp
        self.fluxRcp = ampFluxRcp
        
        lam = scipy.constants.c/(self.freqList*1e3)
        
        self.tempLcp = ampFluxLcp * lam[:, None]**2 * 1e-22 / (2*scipy.constants.k * self.beam_sr[:, None])
        self.tempRcp = ampFluxRcp * lam[:, None]**2 * 1e-22 / (2*scipy.constants.k * self.beam_sr[:, None])
        
        self.visLcp *= self.tempLcp[:, :, None]
        self.visRcp *= self.tempRcp[:, :, None]
        
        self.visLcp *= 2 # flux is divided by 2 for R and L
        self.visRcp *= 2

        self.flux_calibrated = True
            
    def beam(self):
        self.setFrequencyChannel(0)
        self.vis2uv(0, average= 20, PSF = True)
        self.uv2lmImage()
        self.lm2Heliocentric(image_scale = 2)
        arcsecPerPix = self.arcsecPerPixel / 2.
        beam = self.lcp
        contours = (skimage.measure.find_contours(beam, 0.5*beam.max()))[0]
        con = NP.zeros_like(contours)
        con[:,1] = contours[:,0]
        con[:,0] = contours[:,1]
        sunEll = skimage.measure.EllipseModel()
        sunEll.estimate(con)
        major = NP.deg2rad(sunEll.params[2] * arcsecPerPix / 3600.)
        minor = NP.deg2rad(sunEll.params[3] * arcsecPerPix / 3600.)
        self.beam_sr[0] = NP.pi * major * minor / NP.log(2)
        for ff in range(1, self.freqListLength):
            self.beam_sr[ff] = self.beam_sr[0] * (self.freqList[0]/self.freqList[ff])**2
        """
        In original formula (https://science.nrao.edu/facilities/vla/proposing/TBconv)
        theta_maj ang theta_min are full widths of an ellipse, that is why there is
        4 in denominator.
        Here major and minor are semi-axes.
        """
     
    def visIndex2antIndex(self, visIndex):
        if visIndex > self.antennaA.size or visIndex < 0:
            Exception('visIndex is out of range')
        else:
            return self.antennaA[visIndex], self.antennaB[visIndex]
        
    def solarPhase(self, freq):
        u,v,w = base2uvw0306(self.RAO.hAngle, self.RAO.declination, 98, 99)
        baseWave = NP.sqrt(u**2+v**2)*self.freqList[freq]*1e3/constants.c.to_value()
        if baseWave > 120:
            self.nsSolarPhase[freq] = NP.pi
        else:
            self.nsSolarPhase[freq] = 0
        u,v,w = base2uvw0306(self.RAO.hAngle, self.RAO.declination, 1, 2)
        baseWave = NP.sqrt(u**2+v**2)*self.freqList[freq]*1e3/constants.c.to_value()
        if baseWave > 120:
            self.ewSolarPhase[freq] = NP.pi
        else:
            self.ewSolarPhase[freq] = 0

    def calculatePhaseLcp_nonlinear(self, freqChannel):
        redIndexesNS = []
        for baseline in range(1, self.baselines+1):
            redIndexesNS.append(NP.where((self.antennaA==98-1+baseline) & (self.antennaB==33))[0][0])
            for i in range(self.antNumberNS - baseline):
                redIndexesNS.append(NP.where((self.antennaB==98+i) & (self.antennaA==98+i+baseline))[0][0])
    
        redIndexesEW = []
        for baseline in range(1, self.baselines+1):
            for i in range(self.antNumberEW - baseline):
                redIndexesEW.append(NP.where((self.antennaA==1+i) & (self.antennaB==1+i+baseline))[0][0])
             
        validScansBoth = NP.intersect1d(NP.where(self.validScansLcp[freqChannel]), NP.where(self.validScansRcp[freqChannel]))
        ind = NP.argmin(NP.abs(validScansBoth - self.calibIndex))
        calibIndex = validScansBoth[ind]   
             
        if self.averageCalib:
            redundantVisNS = NP.sum(self.visLcp[freqChannel, :20, redIndexesNS], axis = 1)/NP.sum(self.validScansLcp[freqChannel])
            redundantVisEW = NP.sum(self.visLcp[freqChannel, :20, redIndexesEW], axis = 1)/NP.sum(self.validScansLcp[freqChannel])
            redundantVisAll = NP.append(redundantVisEW, redundantVisNS)
        else:
            redundantVisNS = self.visLcp[freqChannel, calibIndex, redIndexesNS]
            redundantVisEW = self.visLcp[freqChannel, calibIndex, redIndexesEW]
            redundantVisAll = NP.append(redundantVisEW, redundantVisNS)
            
        ewAmpSign = 1 if self.ewSolarPhase[freqChannel]==0 else -1
        nsAmpSign = 1 if self.nsSolarPhase[freqChannel]==0 else -1
        
        res = NP.zeros_like(redundantVisAll, dtype = complex)
        ewSolarAmp = 1 * ewAmpSign
        nsAntNumber_c = self.antNumberNS + 1
        nsGainsNumber = self.antNumberNS
        ewGainsNumber = self.antNumberEW
        nsSolVisNumber = self.baselines - 1
        ewSolVisNumber = self.baselines - 1
        nsNum = int((2*self.antNumberNS - (self.baselines-1))/2 * self.baselines)
        ewNum = int((2*(self.antNumberEW-1) - (self.baselines-1))/2 * self.baselines)
        solVisArrayNS = NP.zeros(nsNum, dtype = complex)
        antAGainsNS = NP.zeros(nsNum, dtype = complex)
        antBGainsNS = NP.zeros(nsNum, dtype = complex)
        solVisArrayEW = NP.zeros(ewNum, dtype = complex)
        antAGainsEW = NP.zeros(ewNum, dtype = complex)
        antBGainsEW = NP.zeros(ewNum, dtype = complex)
        ewSolVis = NP.zeros(self.baselines, dtype = complex)
        nsSolVis = NP.zeros(self.baselines, dtype = complex)
        solVis = NP.zeros_like(redundantVisAll, dtype = complex)
        antAGains = NP.zeros_like(redundantVisAll, dtype = complex)
        antBGains = NP.zeros_like(redundantVisAll, dtype = complex)
        
        args = (redundantVisAll, freqChannel,
                res, ewSolarAmp, nsAntNumber_c, nsGainsNumber, ewGainsNumber, nsSolVisNumber, 
                ewSolVisNumber, nsNum, ewNum, solVisArrayNS, antAGainsNS, antBGainsNS, solVisArrayEW, 
                antAGainsEW, antBGainsEW, ewSolVis, nsSolVis, solVis, antAGains, antBGains, nsAmpSign)

        with threadpool_limits(limits=self.n_threads, user_api='blas'):
            ls_res = least_squares(self.allGainsFunc_constrained, self.x_ini_lcp[freqChannel], args = args, max_nfev = 400)
        self.calibrationResultLcp[freqChannel] = ls_res['x']
        gains = srh_utils.real_to_complex(ls_res['x'][1:])[(self.baselines-1)*2:]
        self.ew_gains_lcp = gains[:self.antNumberEW]
        self.ewAntPhaLcp[freqChannel] = NP.angle(self.ew_gains_lcp)
        self.ns_gains_lcp = gains[self.antNumberEW:]
        self.nsAntPhaLcp[freqChannel] = NP.angle(self.ns_gains_lcp)
        
        norm = NP.mean(NP.abs(gains))#[NP.abs(gains)<NP.median(NP.abs(gains))*0.6]))
        self.ewAntAmpLcp[freqChannel] = NP.abs(self.ew_gains_lcp)/norm
        self.ewAntAmpLcp[freqChannel][self.ewAntAmpLcp[freqChannel]<NP.median(self.ewAntAmpLcp[freqChannel])*0.6] = 1e6
        self.nsAntAmpLcp[freqChannel] = NP.abs(self.ns_gains_lcp)/norm
        self.nsAntAmpLcp[freqChannel][self.nsAntAmpLcp[freqChannel]<NP.median(self.nsAntAmpLcp[freqChannel])*0.6] = 1e6
        
        self.calibration_fun_sum_lcp[freqChannel] = NP.sum(ls_res['fun']**2)
        
    def calculatePhaseRcp_nonlinear(self, freqChannel):
        redIndexesNS = []
        for baseline in range(1, self.baselines+1):
            redIndexesNS.append(NP.where((self.antennaA==98-1+baseline) & (self.antennaB==33))[0][0])
            for i in range(self.antNumberNS - baseline):
                redIndexesNS.append(NP.where((self.antennaB==98+i) & (self.antennaA==98+i+baseline))[0][0])
    
        redIndexesEW = []
        for baseline in range(1, self.baselines+1):
            for i in range(self.antNumberEW - baseline):
                redIndexesEW.append(NP.where((self.antennaA==1+i) & (self.antennaB==1+i+baseline))[0][0])
            
        validScansBoth = NP.intersect1d(NP.where(self.validScansLcp[freqChannel]), NP.where(self.validScansRcp[freqChannel]))
        ind = NP.argmin(NP.abs(validScansBoth - self.calibIndex))
        calibIndex = validScansBoth[ind]
        
        if self.averageCalib:
            redundantVisNS = NP.sum(self.visRcp[freqChannel, :20, redIndexesNS], axis = 1)/NP.sum(self.validScansRcp[freqChannel])
            redundantVisEW = NP.sum(self.visRcp[freqChannel, :20, redIndexesEW], axis = 1)/NP.sum(self.validScansRcp[freqChannel])
            redundantVisAll = NP.append(redundantVisEW, redundantVisNS)
        else:
            redundantVisNS = self.visRcp[freqChannel, calibIndex, redIndexesNS]
            redundantVisEW = self.visRcp[freqChannel, calibIndex, redIndexesEW]
            redundantVisAll = NP.append(redundantVisEW, redundantVisNS)
            
        ewAmpSign = 1 if self.ewSolarPhase[freqChannel]==0 else -1
        nsAmpSign = 1 if self.nsSolarPhase[freqChannel]==0 else -1
        
        res = NP.zeros_like(redundantVisAll, dtype = complex)
        ewSolarAmp = 1 * ewAmpSign
        nsAntNumber_c = self.antNumberNS + 1
        nsGainsNumber = self.antNumberNS
        ewGainsNumber = self.antNumberEW
        nsSolVisNumber = self.baselines - 1
        ewSolVisNumber = self.baselines - 1
        nsNum = int((2*self.antNumberNS - (self.baselines-1))/2 * self.baselines)
        ewNum = int((2*(self.antNumberEW-1) - (self.baselines-1))/2 * self.baselines)
        solVisArrayNS = NP.zeros(nsNum, dtype = complex)
        antAGainsNS = NP.zeros(nsNum, dtype = complex)
        antBGainsNS = NP.zeros(nsNum, dtype = complex)
        solVisArrayEW = NP.zeros(ewNum, dtype = complex)
        antAGainsEW = NP.zeros(ewNum, dtype = complex)
        antBGainsEW = NP.zeros(ewNum, dtype = complex)
        ewSolVis = NP.zeros(self.baselines, dtype = complex)
        nsSolVis = NP.zeros(self.baselines, dtype = complex)
        solVis = NP.zeros_like(redundantVisAll, dtype = complex)
        antAGains = NP.zeros_like(redundantVisAll, dtype = complex)
        antBGains = NP.zeros_like(redundantVisAll, dtype = complex)
        
        args = (redundantVisAll, freqChannel,
                res, ewSolarAmp, nsAntNumber_c, nsGainsNumber, ewGainsNumber, nsSolVisNumber, 
                ewSolVisNumber, nsNum, ewNum, solVisArrayNS, antAGainsNS, antBGainsNS, solVisArrayEW, 
                antAGainsEW, antBGainsEW, ewSolVis, nsSolVis, solVis, antAGains, antBGains, nsAmpSign)
        
        with threadpool_limits(limits=self.n_threads, user_api='blas'):
            ls_res = least_squares(self.allGainsFunc_constrained, self.x_ini_rcp[freqChannel], args = args, max_nfev = 400)
        self.calibrationResultRcp[freqChannel] = ls_res['x']
        gains = srh_utils.real_to_complex(ls_res['x'][1:])[(self.baselines-1)*2:]
        self.ew_gains_rcp = gains[:self.antNumberEW]
        self.ewAntPhaRcp[freqChannel] = NP.angle(self.ew_gains_rcp)
        self.ns_gains_rcp = gains[self.antNumberEW:]
        self.nsAntPhaRcp[freqChannel] = NP.angle(self.ns_gains_rcp)
        
        norm = NP.mean(NP.abs(gains))#[NP.abs(gains)<NP.median(NP.abs(gains))*0.6]))
        self.ewAntAmpRcp[freqChannel] = NP.abs(self.ew_gains_rcp)/norm
        self.ewAntAmpRcp[freqChannel][self.ewAntAmpRcp[freqChannel]<NP.median(self.ewAntAmpRcp[freqChannel])*0.6] = 1e6
        self.nsAntAmpRcp[freqChannel] = NP.abs(self.ns_gains_rcp)/norm
        self.nsAntAmpRcp[freqChannel][self.nsAntAmpRcp[freqChannel]<NP.median(self.nsAntAmpRcp[freqChannel])*0.6] = 1e6

        self.calibration_fun_sum_rcp[freqChannel] = NP.sum(ls_res['fun']**2)

    def allGainsFunc_constrained(self, x, obsVis, freq,
                                 res, ewSolarAmp, nsAntNumber_c, nsGainsNumber, ewGainsNumber, nsSolVisNumber, 
                                 ewSolVisNumber, nsNum, ewNum, solVisArrayNS, antAGainsNS, antBGainsNS, solVisArrayEW, 
                                 antAGainsEW, antBGainsEW, ewSolVis, nsSolVis, solVis, antAGains, antBGains, nsAmpSign):

        nsSolarAmp = NP.abs(x[0]) * nsAmpSign
        x_complex = srh_utils.real_to_complex(x[1:])

        ewSolVis[0] = ewSolarAmp
        ewSolVis[1:] = x_complex[: ewSolVisNumber]
        nsSolVis[0] = nsSolarAmp
        nsSolVis[1:] = x_complex[ewSolVisNumber : ewSolVisNumber+nsSolVisNumber]
        
        ewGains = x_complex[ewSolVisNumber+nsSolVisNumber : ewSolVisNumber+nsSolVisNumber+ewGainsNumber]
        nsGains = NP.append(ewGains[32], x_complex[ewSolVisNumber+nsSolVisNumber+ewGainsNumber :])
        
        prev_ind_ns = 0
        prev_ind_ew = 0
        for baseline in range(1, self.baselines+1):
            solVisArrayNS[prev_ind_ns:prev_ind_ns+nsAntNumber_c-baseline] = NP.full(nsAntNumber_c-baseline, nsSolVis[baseline-1])
            antAGainsNS[prev_ind_ns:prev_ind_ns+nsAntNumber_c-baseline] = nsGains[:nsAntNumber_c-baseline]
            antBGainsNS[prev_ind_ns:prev_ind_ns+nsAntNumber_c-baseline] = nsGains[baseline:]
            prev_ind_ns = prev_ind_ns+nsAntNumber_c-baseline
            
            solVisArrayEW[prev_ind_ew:prev_ind_ew+self.antNumberEW-baseline] = NP.full(self.antNumberEW-baseline, ewSolVis[baseline-1])
            antAGainsEW[prev_ind_ew:prev_ind_ew+self.antNumberEW-baseline] = ewGains[:self.antNumberEW-baseline]
            antBGainsEW[prev_ind_ew:prev_ind_ew+self.antNumberEW-baseline] = ewGains[baseline:]
            prev_ind_ew = prev_ind_ew+self.antNumberEW-baseline
            
        solVis[:len(solVisArrayEW)] = solVisArrayEW
        solVis[len(solVisArrayEW):] = solVisArrayNS
        antAGains[:len(antAGainsEW)] = antAGainsEW
        antAGains[len(antAGainsEW):] = antAGainsNS
        antBGains[:len(antBGainsEW)] = antBGainsEW
        antBGains[len(antBGainsEW):] = antBGainsNS
            
        res = solVis * antAGains * NP.conj(antBGains) - obsVis
        return srh_utils.complex_to_real(res)  
    
    def buildEWPhase(self):
        newLcpPhaseCorrection = NP.zeros(self.antNumberEW)
        newRcpPhaseCorrection = NP.zeros(self.antNumberEW)
        for j in range(self.antNumberEW):
                newLcpPhaseCorrection[j] += NP.deg2rad(self.ewSlopeLcp[self.frequencyChannel] * (j - 32)) 
                newRcpPhaseCorrection[j] += NP.deg2rad(self.ewSlopeRcp[self.frequencyChannel] * (j - 32))
        self.ewLcpPhaseCorrection[self.frequencyChannel, :] = newLcpPhaseCorrection[:]
        self.ewRcpPhaseCorrection[self.frequencyChannel, :] = newRcpPhaseCorrection[:]
        
    def buildNSPhase(self):
        newLcpPhaseCorrection = NP.zeros(self.antNumberNS)
        newRcpPhaseCorrection = NP.zeros(self.antNumberNS)
        for j in range(self.antNumberNS):
                newLcpPhaseCorrection[j] += (NP.deg2rad(-self.nsSlopeLcp[self.frequencyChannel] * (j + 1))  + NP.deg2rad(self.nsLcpStair[self.frequencyChannel]))
                newRcpPhaseCorrection[j] += (NP.deg2rad(-self.nsSlopeRcp[self.frequencyChannel] * (j + 1)) + NP.deg2rad(self.nsRcpStair[self.frequencyChannel]))
        self.nsLcpPhaseCorrection[self.frequencyChannel, :] = newLcpPhaseCorrection[:]
        self.nsRcpPhaseCorrection[self.frequencyChannel, :] = newRcpPhaseCorrection[:]
        
    def vis2uv(self, scan, phaseCorrect = True, amplitudeCorrect = False, PSF=False, average = 0):
        self.uvLcp = NP.zeros((self.sizeOfUv, self.sizeOfUv), dtype = complex)
        self.uvRcp = NP.zeros((self.sizeOfUv, self.sizeOfUv), dtype = complex)
        flags_ew = NP.where(self.ewAntAmpLcp[self.frequencyChannel]==1e6)[0]
        flags_ns = NP.where(self.nsAntAmpLcp[self.frequencyChannel]==1e6)[0]
        
        ewPhLcp = self.ewAntPhaLcp[self.frequencyChannel] + self.ewLcpPhaseCorrection[self.frequencyChannel]
        nsPhLcp = self.nsAntPhaLcp[self.frequencyChannel] + self.nsLcpPhaseCorrection[self.frequencyChannel]
        ewAmpLcp = self.ewAntAmpLcp[self.frequencyChannel]
        nsAmpLcp = self.nsAntAmpLcp[self.frequencyChannel]
        if self.useRLDif:
            ewPhRcp = ewPhLcp - self.ewPhaseDif[self.frequencyChannel]
            nsPhRcp = nsPhLcp - self.nsPhaseDif[self.frequencyChannel]
            ewAmpRcp = ewAmpLcp/self.ewAmpDif[self.frequencyChannel]
            nsAmpRcp = nsAmpLcp/self.nsAmpDif[self.frequencyChannel]
        else:
            ewPhRcp = self.ewAntPhaRcp[self.frequencyChannel] + self.ewRcpPhaseCorrection[self.frequencyChannel]
            nsPhRcp = self.nsAntPhaRcp[self.frequencyChannel] + self.nsRcpPhaseCorrection[self.frequencyChannel]
            ewAmpRcp = self.ewAntAmpRcp[self.frequencyChannel]
            nsAmpRcp = self.nsAntAmpRcp[self.frequencyChannel]
        
        O = self.sizeOfUv//2
        if average:
            firstScan = scan
            if  self.visLcp.shape[1] < (scan + average):
                lastScan = self.dataLength
            else:
                lastScan = scan + average
            for i in range(self.antNumberNS):
                for j in range(self.antNumberEW):
                    if not (NP.any(flags_ew == j) or NP.any(flags_ns == i)):
                        self.uvLcp[O + (i+1)*2, O + (j-32)*2] = NP.sum(self.visLcp[self.frequencyChannel, firstScan:lastScan, i*97+j])/NP.sum(self.validScansLcp[self.frequencyChannel][firstScan:lastScan])
                        self.uvRcp[O + (i+1)*2, O + (j-32)*2] = NP.sum(self.visRcp[self.frequencyChannel, firstScan:lastScan, i*97+j])/NP.sum(self.validScansRcp[self.frequencyChannel][firstScan:lastScan])
                        if (phaseCorrect):
                            self.uvLcp[O + (i+1)*2, O + (j-32)*2] *= NP.exp(1j * (-ewPhLcp[j] + nsPhLcp[i]))
                            self.uvRcp[O + (i+1)*2, O + (j-32)*2] *= NP.exp(1j * (-ewPhRcp[j] + nsPhRcp[i]))
                        if (amplitudeCorrect):
                            self.uvLcp[O + (i+1)*2, O + (j-32)*2] /= (ewAmpLcp[j] * nsAmpLcp[i])
                            self.uvRcp[O + (i+1)*2, O + (j-32)*2] /= (ewAmpRcp[j] * nsAmpRcp[i])
                        self.uvLcp[O - (i+1)*2, O - (j-32)*2] = NP.conj(self.uvLcp[O + (i+1)*2, O + (j-32)*2])
                        self.uvRcp[O - (i+1)*2, O - (j-32)*2] = NP.conj(self.uvRcp[O + (i+1)*2, O + (j-32)*2])
            for i in range(self.antNumberEW):
                if not (NP.any(flags_ew == i) or NP.any(flags_ew == 32)):
                    if i<32:
                        self.uvLcp[O, O + (i-32)*2] = NP.sum(self.visLcp[self.frequencyChannel, firstScan:lastScan, self.antZeroRow[i]])/NP.sum(self.validScansLcp[self.frequencyChannel][firstScan:lastScan])
                        self.uvRcp[O, O + (i-32)*2] = NP.sum(self.visRcp[self.frequencyChannel, firstScan:lastScan, self.antZeroRow[i]])/NP.sum(self.validScansRcp[self.frequencyChannel][firstScan:lastScan])
                        if (phaseCorrect):
                            self.uvLcp[O, O + (i-32)*2] *= NP.exp(1j * (-ewPhLcp[i] + ewPhLcp[32]))
                            self.uvRcp[O, O + (i-32)*2] *= NP.exp(1j * (-ewPhRcp[i] + ewPhRcp[32]))
                        if (amplitudeCorrect):
                            self.uvLcp[O, O + (i-32)*2] /= (ewAmpLcp[i] * ewAmpLcp[32])
                            self.uvRcp[O, O + (i-32)*2] /= (ewAmpRcp[i] * ewAmpRcp[32])
    #                    self.uvLcp[O, O + (32-i)*2] = NP.conj(self.uvLcp[O, O + (i-32)*2])
    #                    self.uvRcp[O, O + (32-i)*2] = NP.conj(self.uvRcp[O, O + (i-32)*2])
                    if i>32:
                        self.uvLcp[O, O + (i-32)*2] = NP.conj(NP.sum(self.visLcp[self.frequencyChannel, firstScan:lastScan, self.antZeroRow[i]]))/NP.sum(self.validScansLcp[self.frequencyChannel][firstScan:lastScan])
                        self.uvRcp[O, O + (i-32)*2] = NP.conj(NP.sum(self.visRcp[self.frequencyChannel, firstScan:lastScan, self.antZeroRow[i]]))/NP.sum(self.validScansRcp[self.frequencyChannel][firstScan:lastScan])
                        if (phaseCorrect):
                            self.uvLcp[O, O + (i-32)*2] *= NP.exp(1j * (-ewPhLcp[i] + ewPhLcp[32]))
                            self.uvRcp[O, O + (i-32)*2] *= NP.exp(1j * (-ewPhRcp[i] + ewPhRcp[32]))
                        if (amplitudeCorrect):
                            self.uvLcp[O, O + (i-32)*2] /= (ewAmpLcp[i] * ewAmpLcp[32])
                            self.uvRcp[O, O + (i-32)*2] /= (ewAmpRcp[i] * ewAmpRcp[32])
    #                    self.uvLcp[O, O + (32-i)*2] = NP.conj(self.uvLcp[O, O + (i-32)*2])
    #                    self.uvRcp[O, O + (32-i)*2] = NP.conj(self.uvRcp[O, O + (i-32)*2])
                
        else:
            for i in range(31):
                for j in range(self.antNumberEW):
                    if not (NP.any(flags_ew == j) or NP.any(flags_ns == i)):
                        self.uvLcp[O + (i+1)*2, O + (j-32)*2] = self.visLcp[self.frequencyChannel, scan, i*97+j]
                        self.uvRcp[O + (i+1)*2, O + (j-32)*2] = self.visRcp[self.frequencyChannel, scan, i*97+j]
                        if (phaseCorrect):
                            self.uvLcp[O + (i+1)*2, O + (j-32)*2] *= NP.exp(1j * (-ewPhLcp[j] + nsPhLcp[i]))
                            self.uvRcp[O + (i+1)*2, O + (j-32)*2] *= NP.exp(1j * (-ewPhRcp[j] + nsPhRcp[i]))
                        if (amplitudeCorrect):
                            self.uvLcp[O + (i+1)*2, O + (j-32)*2] /= (ewAmpLcp[j] * nsAmpLcp[i])
                            self.uvRcp[O + (i+1)*2, O + (j-32)*2] /= (ewAmpRcp[j] * nsAmpRcp[i])
                        self.uvLcp[O - (i+1)*2, O - (j-32)*2] = NP.conj(self.uvLcp[O + (i+1)*2, O + (j-32)*2])
                        self.uvRcp[O - (i+1)*2, O - (j-32)*2] = NP.conj(self.uvRcp[O + (i+1)*2, O + (j-32)*2])
                    
            for i in range(self.antNumberEW):
                if not (NP.any(flags_ew == i) or NP.any(flags_ew == 32)):
                    if i<32:
                        self.uvLcp[O, O + (i-32)*2] = self.visLcp[self.frequencyChannel, scan, self.antZeroRow[i]]
                        self.uvRcp[O, O + (i-32)*2] = self.visRcp[self.frequencyChannel, scan, self.antZeroRow[i]]
                        if (phaseCorrect):
                            self.uvLcp[O, O + (i-32)*2] *= NP.exp(1j * (-ewPhLcp[i] + ewPhLcp[32]))
                            self.uvRcp[O, O + (i-32)*2] *= NP.exp(1j * (-ewPhRcp[i] + ewPhRcp[32]))
                        if (amplitudeCorrect):
                            self.uvLcp[O, O + (i-32)*2] /= (ewAmpLcp[i] * ewAmpLcp[32])
                            self.uvRcp[O, O + (i-32)*2] /= (ewAmpRcp[i] * ewAmpRcp[32])
                        self.uvLcp[O, O + (32-i)*2] = NP.conj(self.uvLcp[O, O + (i-32)*2])
    #                    self.uvRcp[O, O + (32-i)*2] = NP.conj(self.uvRcp[O, O + (i-32)*2])
                    if i>32:
                        self.uvLcp[O, O + (i-32)*2] = NP.conj(self.visLcp[self.frequencyChannel, scan, self.antZeroRow[i]])
                        self.uvRcp[O, O + (i-32)*2] = NP.conj(self.visRcp[self.frequencyChannel, scan, self.antZeroRow[i]])
                        if (phaseCorrect):
                            self.uvLcp[O, O + (i-32)*2] *= NP.exp(1j * (-ewPhLcp[i] + ewPhLcp[32]))
                            self.uvRcp[O, O + (i-32)*2] *= NP.exp(1j * (-ewPhRcp[i] + ewPhRcp[32]))
                        if (amplitudeCorrect):
                            self.uvLcp[O, O + (i-32)*2] /= (ewAmpLcp[i] * ewAmpLcp[32])
                            self.uvRcp[O, O + (i-32)*2] /= (ewAmpRcp[i] * ewAmpRcp[32])
                        # self.uvLcp[O, O + (32-i)*2] = NP.conj(self.uvLcp[O, O + (i-32)*2])
                        # self.uvRcp[O, O + (32-i)*2] = NP.conj(self.uvRcp[O, O + (i-32)*2])
        if (amplitudeCorrect):
            if average:
                self.uvLcp[O,O] = NP.mean(self.fluxLcp[self.frequencyChannel, firstScan:lastScan])*2
                self.uvRcp[O,O] = NP.mean(self.fluxRcp[self.frequencyChannel, firstScan:lastScan])*2
            else:
                self.uvLcp[O,O] = self.fluxLcp[self.frequencyChannel, scan]*2
                self.uvRcp[O,O] = self.fluxRcp[self.frequencyChannel, scan]*2
        
        if PSF:
            self.uvLcp[NP.abs(self.uvLcp)>1e-8] = 1
            self.uvRcp[NP.abs(self.uvRcp)>1e-8] = 1
            
        self.uvLcp[NP.abs(self.uvLcp)<1e-6] = 0.
        self.uvRcp[NP.abs(self.uvRcp)<1e-6] = 0.
        self.uvLcp /= NP.count_nonzero(self.uvLcp)
        self.uvRcp /= NP.count_nonzero(self.uvRcp)
        
    def uv2lmImage(self):
        self.lcp = NP.fft.fft2(NP.roll(NP.roll(self.uvLcp,self.sizeOfUv//2+1,0),self.sizeOfUv//2+1,1));
        self.lcp = NP.roll(NP.roll(self.lcp,self.sizeOfUv//2-1,0),self.sizeOfUv//2-1,1);
        self.lcp = NP.flip(self.lcp, 1)
        self.rcp = NP.fft.fft2(NP.roll(NP.roll(self.uvRcp,self.sizeOfUv//2+1,0),self.sizeOfUv//2+1,1));
        self.rcp = NP.roll(NP.roll(self.rcp,self.sizeOfUv//2-1,0),self.sizeOfUv//2-1,1);
        self.rcp = NP.flip(self.rcp, 1)
        
    def lm2Heliocentric(self, image_scale = 0.5):
        scaling = self.RAO.getPQScale(self.sizeOfUv, NP.deg2rad(self.arcsecPerPixel * (self.sizeOfUv - 1)/3600.)/image_scale, self.freqList[self.frequencyChannel]*1e3)
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

    def createDiskLmFft(self, radius, arcsecPerPixel = 2.45552):
        scaling = self.RAO.getPQScale(self.sizeOfUv, NP.deg2rad(arcsecPerPixel*(self.sizeOfUv-1)/3600.), self.freqList[self.frequencyChannel]*1e3)
        scale = AffineTransform(scale=(scaling[0]/self.sizeOfUv, scaling[1]/self.sizeOfUv))
        back_shift = AffineTransform(translation=(self.sizeOfUv/2, self.sizeOfUv/2))
        shift = AffineTransform(translation=(-self.sizeOfUv/2, -self.sizeOfUv/2))
        matrix = AffineTransform(matrix = NP.linalg.inv(self.RAO.getPQ2HDMatrix()))

        qSun = srh_utils.createDisk(self.sizeOfUv, radius, arcsecPerPixel)
        
        dL = 2*( 30//2) + 1
        arg_x = NP.linspace(-1.,1,dL)
        arg_y = NP.linspace(-1.,1,dL)
        xx, yy = NP.meshgrid(arg_x, arg_y)

        gKern =   NP.exp(-0.5*(xx**2 + yy**2))
        qSmoothSun = scipy.signal.fftconvolve(qSun,gKern) / dL**2
        qSmoothSun = qSmoothSun[dL//2:dL//2+self.sizeOfUv,dL//2:dL//2+self.sizeOfUv]
        smoothCoef = qSmoothSun[512, 512]
        qSmoothSun /= smoothCoef
        
        res = warp(qSmoothSun, (shift + (matrix + back_shift)).inverse)
        qSun_lm = warp(res,(shift + (scale + back_shift)).inverse)
        qSun_lm_fft = NP.fft.fft2(NP.roll(NP.roll(qSun_lm,self.sizeOfUv//2,0),self.sizeOfUv//2,1));
        qSun_lm_fft = NP.roll(NP.roll(qSun_lm_fft,self.sizeOfUv//2,0),self.sizeOfUv//2,1)# / self.sizeOfUv;
        qSun_lm_fft = NP.flip(qSun_lm_fft, 0)
#        qSun_lm_uv = qSun_lm_fft * uvPsf
#        qSun_lm_conv = NP.fft.fft2(NP.roll(NP.roll(qSun_lm_uv,self.sizeOfUv//2+1,0),self.sizeOfUv//2+1,1));
#        qSun_lm_conv = NP.roll(NP.roll(qSun_lm_conv,self.sizeOfUv//2-1,0),self.sizeOfUv//2-1,1);
#        qSun_lm_conv = NP.flip(NP.flip(qSun_lm_conv, 1), 0)
        self.lm_hd_relation[self.frequencyChannel] = NP.sum(qSun_lm)/NP.sum(qSmoothSun)
        self.fftDisk = qSun_lm_fft #qSun_lm_conv, 
    
    def createUvUniform(self):
        self.uvUniform = NP.zeros((self.sizeOfUv, self.sizeOfUv), dtype = complex)
        flags_ew = NP.where(self.ewAntAmpLcp[self.frequencyChannel]==1e6)[0]
        flags_ns = NP.where(self.nsAntAmpLcp[self.frequencyChannel]==1e6)[0]
        O = self.sizeOfUv//2
        for i in range(self.antNumberNS):
            for j in range(self.antNumberEW):
                if not (NP.any(flags_ew == j) or NP.any(flags_ns == i)):
                    self.uvUniform[O + (i+1)*2, O + (j-32)*2] = 1
                    self.uvUniform[O - (i+1)*2, O - (j-32)*2] = 1
        for i in range(self.antNumberEW):
            if i != 32:
                if not (NP.any(flags_ew == i) or NP.any(flags_ew == 32)):
                    self.uvUniform[O, O + (i-32)*2] = 1
        self.uvUniform[O, O] = 1
        self.uvUniform /= NP.count_nonzero(self.uvUniform)
                    
    def createUvPsf(self, T, ewSlope, nsSlope, shift):
        self.uvPsf = self.uvUniform.copy()
        O = self.sizeOfUv//2
        ewSlope = NP.deg2rad(ewSlope)
        nsSlope = NP.deg2rad(nsSlope)
        ewSlopeUv = NP.linspace(-O * ewSlope/2., O * ewSlope/2., self.sizeOfUv)
        nsSlopeUv = NP.linspace(-O * nsSlope/2., O * nsSlope/2., self.sizeOfUv)
        ewGrid,nsGrid = NP.meshgrid(ewSlopeUv, nsSlopeUv)
        slopeGrid = ewGrid + nsGrid
        slopeGrid[self.uvUniform == 0] = 0
        self.uvPsf *= T * NP.exp(1j * slopeGrid)
        self.uvPsf[O,O] = shift/NP.count_nonzero(self.uvUniform)
    
    def diskDiff(self, x, pol):
        self.createUvPsf(x[0], x[1], x[2], x[3])
        uvDisk = self.fftDisk * self.uvPsf
        if pol == 0:
            diff = self.uvLcp - uvDisk
        if pol == 1:
            diff = self.uvRcp - uvDisk
        return srh_utils.complex_to_real(diff[self.uvUniform!=0])
#        qSun_lm_conv = NP.fft.fft2(NP.roll(NP.roll(diff,uvSize//2+1,0),uvSize//2+1,1));
#        return NP.abs(NP.reshape(qSun_lm_conv, uvSize**2))
    
    def findDisk(self):
        Tb = self.ZirinQSunTb.getTbAtFrequency(self.freqList[self.frequencyChannel]*1e-6) * 1e3
        self.createDiskLmFft(980)
        self.createUvUniform()
        self.x_ini = [Tb/self.convolutionNormCoef,0,0,1]
        # x_ini = [1,0,0]
        with threadpool_limits(limits=self.n_threads, user_api='blas'):
            self.center_ls_res_lcp = least_squares(self.diskDiff, self.x_ini, args = (0,))
            _diskLevelLcp, _ewSlopeLcp, _nsSlopeLcp, _shiftLcp = self.center_ls_res_lcp['x']
            self.center_ls_res_rcp = least_squares(self.diskDiff, self.x_ini, args = (1,))
            _diskLevelRcp, _ewSlopeRcp, _nsSlopeRcp, _shiftRcp = self.center_ls_res_rcp['x']
        
        self.diskLevelLcp[self.frequencyChannel] = _diskLevelLcp
        self.diskLevelRcp[self.frequencyChannel] = _diskLevelRcp

        self.lcpShift[self.frequencyChannel] = self.lcpShift[self.frequencyChannel]/(_shiftLcp * self.convolutionNormCoef / Tb)
        self.rcpShift[self.frequencyChannel] = self.rcpShift[self.frequencyChannel]/(_shiftRcp * self.convolutionNormCoef / Tb)

        if not self.corr_amp_exist:
            self.ewAntAmpLcp[self.frequencyChannel][self.ewAntAmpLcp[self.frequencyChannel]!=1e6] *= NP.sqrt(_diskLevelLcp*self.convolutionNormCoef / Tb)
            self.nsAntAmpLcp[self.frequencyChannel][self.nsAntAmpLcp[self.frequencyChannel]!=1e6] *= NP.sqrt(_diskLevelLcp*self.convolutionNormCoef / Tb)
            self.ewAntAmpRcp[self.frequencyChannel][self.ewAntAmpRcp[self.frequencyChannel]!=1e6] *= NP.sqrt(_diskLevelRcp*self.convolutionNormCoef / Tb)
            self.nsAntAmpRcp[self.frequencyChannel][self.nsAntAmpRcp[self.frequencyChannel]!=1e6] *= NP.sqrt(_diskLevelRcp*self.convolutionNormCoef / Tb)

        self.ewSlopeLcp[self.frequencyChannel] = srh_utils.wrap(self.ewSlopeLcp[self.frequencyChannel] + _ewSlopeLcp)
        self.nsSlopeLcp[self.frequencyChannel] = srh_utils.wrap(self.nsSlopeLcp[self.frequencyChannel] + _nsSlopeLcp)
        self.ewSlopeRcp[self.frequencyChannel] = srh_utils.wrap(self.ewSlopeRcp[self.frequencyChannel] + _ewSlopeRcp)
        self.nsSlopeRcp[self.frequencyChannel] = srh_utils.wrap(self.nsSlopeRcp[self.frequencyChannel] + _nsSlopeRcp)

    def correctPhaseSlopeRL(self, freq):
        workingAnts_ew = NP.arange(0,97,1)
        workingAnts_ew = NP.delete(workingAnts_ew, NP.append(self.flags_ew, NP.array((28,))))
        phaseDif_ew = NP.unwrap((self.ewAntPhaLcp[freq][workingAnts_ew]+self.ewLcpPhaseCorrection[freq][workingAnts_ew])
                             - (self.ewAntPhaRcp[freq][workingAnts_ew]+self.ewRcpPhaseCorrection[freq][workingAnts_ew]))
        A = NP.vstack([workingAnts_ew, NP.ones(len(workingAnts_ew))]).T
        ew_slope, c = NP.linalg.lstsq(A, phaseDif_ew, rcond=None)[0]
        workingAnts_ns = NP.arange(0,31,1)
        workingAnts_ns = NP.delete(workingAnts_ns, self.flags_ns)
        phaseDif_ns = NP.unwrap((self.nsAntPhaLcp[freq][workingAnts_ns]+self.nsLcpPhaseCorrection[freq][workingAnts_ns])
                             - (self.nsAntPhaRcp[freq][workingAnts_ns]+self.nsRcpPhaseCorrection[freq][workingAnts_ns]))
        A = NP.vstack([workingAnts_ns, NP.ones(len(workingAnts_ns))]).T
        ns_slope, c = NP.linalg.lstsq(A, phaseDif_ns, rcond=None)[0]
        self.ewSlopeRcp[freq] = srh_utils.wrap(self.ewSlopeRcp[freq] + NP.rad2deg(ew_slope))
        self.nsSlopeRcp[freq] = srh_utils.wrap(self.nsSlopeRcp[freq] - NP.rad2deg(ns_slope))

    def centerDisk(self):
        self.findDisk()
        self.buildEWPhase()
        self.buildNSPhase()
        self.correctPhaseSlopeRL(self.frequencyChannel)
        self.buildEWPhase()
        self.buildNSPhase()
        
    def modelDiskConv(self):
        # self.createUvPsf(self.diskLevelLcp,0,0,0)
        currentDiskTb = self.ZirinQSunTb.getTbAtFrequency(self.freqList[self.frequencyChannel]*1e-6)*1e3
        self.createUvPsf(currentDiskTb/self.convolutionNormCoef,0,0,currentDiskTb/self.convolutionNormCoef)
        self.uvDiskConv = self.fftDisk * self.uvPsf# - self.uvLcp
        qSun_lm = NP.fft.fft2(NP.roll(NP.roll(self.uvDiskConv,self.sizeOfUv//2+1,0),self.sizeOfUv//2+1,1));
        qSun_lm = NP.roll(NP.roll(qSun_lm,self.sizeOfUv//2-1,0),self.sizeOfUv//2-1,1)# / self.sizeOfUv;
        self.modelDisk = qSun_lm
        
    def modelDiskConv_unity(self):
        self.createDiskLmFft(980)
        self.createUvUniform()
        self.createUvPsf(1,0,0,1)
        self.uvDiskConv = self.fftDisk * self.uvPsf# - self.uvLcp
        qSun_lm = NP.fft.fft2(NP.roll(NP.roll(self.uvDiskConv,self.sizeOfUv//2+1,0),self.sizeOfUv//2+1,1));
        qSun_lm = NP.roll(NP.roll(qSun_lm,self.sizeOfUv//2-1,0),self.sizeOfUv//2-1,1)# / self.sizeOfUv;
        qSun_lm = NP.flip(qSun_lm, 0)
        self.modelDisk = qSun_lm
        
    def saveAsUvFits(self, filename, **kwargs):
        uv_fits = SrhUVData()
        uv_fits.write_uvfits_0306(self, filename, **kwargs)
    
    def clean(self, imagename = 'images/0', cell = 2.45, imsize = 1024, niter = 100000, threshold = 60000, stokes = 'RRLL', **kwargs):
        tclean(vis = self.ms_name,
               imagename = imagename,
               cell = cell, 
               imsize = imsize,
               niter = niter,
               threshold = threshold,
               stokes = stokes,
               **kwargs)
        
    def makeMask(self, maskname = 'images/mask', cell = 2.45, imsize = 1024, threshold=40000, stokes = 'RRLL', **kwargs):
        freq_current = self.frequencyChannel
        self.setFrequencyChannel(0)
        if NP.count_nonzero(self.ewAntPhaLcp[0])==0:
            self.calibrate(freq = 0)
        self.saveAsUvFits(maskname + '.fits')
        self.MSfromUvFits(maskname + '.fits', maskname + '_temp.ms')
        os.system('rm \"' + maskname + '.fits\"')
        tclean(vis = maskname + '_temp.ms',
               imagename = maskname + '_temp',
               cell = cell, 
               imsize = imsize,
               niter = 10000,
               threshold=threshold,
               stokes = stokes,
               **kwargs)
        
        ia = IA()
        self.mask_name = maskname
        os.system('cp -r \"%s_temp.model\" \"%s\"' % (maskname, maskname))
        rmtables(tablenames = '%s_temp.*' % maskname)
        
        ia.open(self.mask_name)
        ia_data = ia.getchunk()
        mask0 = ia_data[:,:,0,0]#.transpose()
        mask1 = ia_data[:,:,1,0]#.transpose()

        dL = 2*(50//2) + 1
        arg_x = NP.linspace(-1.,1,dL)
        arg_y = NP.linspace(-1.,1,dL)
        xx, yy = NP.meshgrid(arg_x, arg_y)
        gKern =   NP.exp(-0.5*(xx**2 + yy**2))
        smooth_mask0 = scipy.signal.fftconvolve(mask0,gKern) / dL**2
        smooth_mask0 = smooth_mask0[dL//2:dL//2+imsize,dL//2:dL//2+imsize]
        smooth_mask_cut0 = (smooth_mask0 > 100).astype(int)
        smooth_mask1 = scipy.signal.fftconvolve(mask1,gKern) / dL**2
        smooth_mask1 = smooth_mask1[dL//2:dL//2+imsize,dL//2:dL//2+imsize]
        smooth_mask_cut1 = (smooth_mask1 > 100).astype(int)
        
        ia_data_new = NP.zeros_like(ia_data)
        ia_data_new[:,:,0,0] = smooth_mask_cut0
        ia_data_new[:,:,1,0] = smooth_mask_cut1
        ia.putchunk(pixels=ia_data_new)
        ia.unlock()
        ia.close()
        
        self.setFrequencyChannel(freq_current)

    def makeModel(self, modelname = 'images/model', imagename = 'images/temp', cell = 2.45, imsize = 1024, threshold=100000, stokes = 'RRLL', **kwargs):
        tclean(vis = self.ms_name,
               imagename = imagename,
               cell = cell, 
               imsize = imsize,
               niter = 0,
               threshold=threshold,
               stokes = stokes,
               **kwargs)
        
        ia = IA()
        self.model_name = modelname
        os.system('cp -r \"%s.model\" \"%s\"' % (imagename, self.model_name))
        
        ia.open(imagename + '.image')
        self.restoring_beam = ia.restoringbeam()['beams']['*0']['*0']
        ia.close()
        
        rmtables(tablenames = '%s.*' % imagename)
        
        ia.open(self.model_name)
        ia_data = ia.getchunk()
        diskTb = self.ZirinQSunTb.getTbAtFrequency(self.freqList[self.frequencyChannel]*1e-6) * 1e3 
        disk = srh_utils.createDisk(self.sizeOfUv, arcsecPerPixel = cell)
        dL = 2*( 10//2) + 1
        kern = NP.ones((dL,dL))
        disk_model = scipy.signal.fftconvolve(disk,kern) / dL**2
        disk_model = disk_model[dL//2:dL//2+imsize,dL//2:dL//2+imsize]
        disk_model[disk_model<1e-10] = 0
        disk_model = disk_model * diskTb * self.lm_hd_relation[self.frequencyChannel] / self.convolutionNormCoef# / 4
        ia_data[:,:,0,0] = disk_model
        ia_data[:,:,1,0] = disk_model
        ia.putchunk(pixels=ia_data)
        ia.unlock()
        ia.close()
        
    def casaImage2Fits(self, casa_imagename, fits_imagename, cell, imsize, naxis, scan, RL = False, save_model = False, clean_disk = True):
        ia = IA()
        ia.open(casa_imagename + '.image')
        try:
            restoring_beam = ia.restoringbeam()['beams']['*0']['*0']
        except:
            restoring_beam = ia.restoringbeam()
        rcp = ia.getchunk()[:,:,0,0].transpose()
        lcp = ia.getchunk()[:,:,1,0].transpose()
        ia.close()

        shift = AffineTransform(translation=(-imsize/2,-imsize/2))
        rotate = AffineTransform(rotation = -self.RAO.pAngle)
        back_shift = AffineTransform(translation=(imsize/2,imsize/2))
        
        rcp = warp(rcp,(shift + (rotate + back_shift)).inverse)
        lcp = warp(lcp,(shift + (rotate + back_shift)).inverse)
        
        if not clean_disk:
            rcp[rcp!=0] += self.tempRcp[self.frequencyChannel, scan]*2 /NP.count_nonzero(self.uvRcp)
            lcp[lcp!=0] += self.tempLcp[self.frequencyChannel, scan]*2 /NP.count_nonzero(self.uvLcp)
   
        if imsize != naxis:
            x, y = NP.linspace(0,imsize-1,imsize), NP.linspace(0,imsize-1,imsize)
            interp = RegularGridInterpolator((x, y), rcp)
            xx,yy = NP.linspace(0,imsize-1,naxis), NP.linspace(0,imsize-1,naxis)
            X, Y = NP.meshgrid(xx, yy, indexing='ij')
            rcp = interp((X, Y))
            interp = RegularGridInterpolator((x, y), lcp)
            lcp = interp((X, Y))
            cdelt = cell * imsize / naxis
            crpix = naxis/2
        else:
            cdelt = cell
            crpix = imsize/2
        
        a,b,ang = restoring_beam['major']['value'],restoring_beam['minor']['value'],restoring_beam['positionangle']['value']
        fitsTime = srh_utils.ihhmm_format(self.freqTime[self.frequencyChannel, scan])
    
        pHeader = fits.Header();
        pHeader['DATE-OBS']     = self.hduList[0].header['DATE-OBS']+'T'+fitsTime
        pHeader['T-OBS']        = fitsTime
        pHeader['INSTRUME']     = self.hduList[0].header['INSTRUME']
        pHeader['ORIGIN']       = self.hduList[0].header['ORIGIN']
        pHeader['FREQUENC']     = ('%d') % (self.freqList[self.frequencyChannel]/1e3 + 0.5)
        pHeader['NAXIS']        = 2
        pHeader['NAXIS1']       = naxis
        pHeader['NAXIS2']       = naxis
        pHeader['CDELT1']       = cdelt
        pHeader['CDELT2']       = cdelt
        pHeader['CRPIX1']       = crpix
        pHeader['CRPIX2']       = crpix
        pHeader['CTYPE1']       = 'HPLN-TAN'
        pHeader['CTYPE2']       = 'HPLT-TAN'
        pHeader['CUNIT1']       = 'arcsec'
        pHeader['CUNIT2']       = 'arcsec'
        pHeader['PSF_ELLA']     = a # PSF ellipse A arcsec
        pHeader['PSF_ELLB']     = b # PSF ellipse B arcsec
        pHeader['PSF_ELLT']     = ang # PSF ellipse theta deg
        pHeader['BSCALE'] = 1
        pHeader['BZERO'] = 0
        pHeader['CRVAL1'] = 0
        pHeader['CRVAL2'] = 0
        pHeader['WAVELNTH'] = pHeader['FREQUENC'] + ' MHz'
        
        date = self.hduList[0].header['DATE-OBS']+'T'+fitsTime
        observer = get_earth(date)
        header = make_heliographic_header(date, observer, [512, 512], frame='carrington')
        pHeader['HGLT_OBS'] = header['hglt_obs']
        pHeader['DSUN_OBS'] = header['dsun_obs']
        pHeader['HGLN_OBS'] = header['hgln_obs']
        pHeader['RSUN_REF'] = header['rsun_ref']
        
        ewLcpPhaseColumn = fits.Column(name='ewLcpPhase', format='D', array = self.ewAntPhaLcp[self.frequencyChannel,:] + self.ewLcpPhaseCorrection[self.frequencyChannel,:])
        ewRcpPhaseColumn = fits.Column(name='ewRcpPhase', format='D', array = self.ewAntPhaRcp[self.frequencyChannel,:] + self.ewRcpPhaseCorrection[self.frequencyChannel,:])
        nsLcpPhaseColumn = fits.Column(name='nsLcpPhase',   format='D', array = self.nsAntPhaLcp[self.frequencyChannel,:] + self.nsLcpPhaseCorrection[self.frequencyChannel,:])
        nsRcpPhaseColumn = fits.Column(name='nsRcpPhase',   format='D', array = self.nsAntPhaRcp[self.frequencyChannel,:] + self.nsRcpPhaseCorrection[self.frequencyChannel,:])
        saveFitsIExtHdu = fits.BinTableHDU.from_columns([ewLcpPhaseColumn, ewRcpPhaseColumn, nsLcpPhaseColumn, nsRcpPhaseColumn])
        if RL:
            saveFitsRCPhdu = fits.PrimaryHDU(header=pHeader, data=rcp.astype('float32'))
            saveFitsRCPpath = fits_imagename + '_RCP.fit'
            hduList = fits.HDUList([saveFitsRCPhdu, saveFitsIExtHdu])
            hduList.writeto(saveFitsRCPpath, overwrite=True)
            
            saveFitsLCPhdu = fits.PrimaryHDU(header=pHeader, data=lcp.astype('float32'))
            saveFitsLCPpath = fits_imagename + '_LCP.fit'
            hduList = fits.HDUList(saveFitsLCPhdu)
            hduList.writeto(saveFitsLCPpath, overwrite=True)
            
            self.out_filenames.append(saveFitsRCPpath)
            self.out_filenames.append(saveFitsLCPpath)
            
        else:
            iImage = (rcp + lcp)/2
            vImage = (rcp - lcp)/2
            saveFitsIhdu = fits.PrimaryHDU(header=pHeader, data=iImage.astype('float32'))
            saveFitsIpath = fits_imagename + '_I.fit'
            hduList = fits.HDUList([saveFitsIhdu, saveFitsIExtHdu])
            hduList.writeto(saveFitsIpath, overwrite=True)
            
            saveFitsVhdu = fits.PrimaryHDU(header=pHeader, data=vImage.astype('float32'))
            saveFitsVpath = fits_imagename + '_V.fit'
            hduList = fits.HDUList(saveFitsVhdu)
            hduList.writeto(saveFitsVpath, overwrite=True)
            
            self.out_filenames.append(saveFitsIpath)
            self.out_filenames.append(saveFitsVpath)
            
        if save_model:
            ia.open(casa_imagename + '.model')
            rcp_model = ia.getchunk()[:,:,0,0].transpose()
            lcp_model = ia.getchunk()[:,:,1,0].transpose()
            ia.close()

            rcp_model = warp(rcp_model,(shift + (rotate + back_shift)).inverse)
            lcp_model = warp(lcp_model,(shift + (rotate + back_shift)).inverse)
       
            if imsize != naxis:
                x, y = NP.linspace(0,imsize-1,imsize), NP.linspace(0,imsize-1,imsize)
                interp = RegularGridInterpolator((x, y), rcp_model)
                xx,yy = NP.linspace(0,imsize-1,naxis), NP.linspace(0,imsize-1,naxis)
                X, Y = NP.meshgrid(xx, yy, indexing='ij')
                rcp_model = interp((X, Y))
                interp = RegularGridInterpolator((x, y), lcp_model)
                lcp_model = interp((X, Y))
            
            saveFitsRCPhdu = fits.PrimaryHDU(header=pHeader, data=rcp_model.astype('float32'))
            saveFitsRCPpath = fits_imagename + '_RCP_model.fit'
            hduList = fits.HDUList([saveFitsRCPhdu, saveFitsIExtHdu])
            hduList.writeto(saveFitsRCPpath, overwrite=True)
            
            saveFitsLCPhdu = fits.PrimaryHDU(header=pHeader, data=lcp_model.astype('float32'))
            saveFitsLCPpath = fits_imagename + '_LCP_model.fit'
            hduList = fits.HDUList(saveFitsLCPhdu)
            hduList.writeto(saveFitsLCPpath, overwrite=True)
            
            self.out_filenames.append(saveFitsRCPpath)
            self.out_filenames.append(saveFitsLCPpath)
        
        
    def makeImage(self, path = './', calibtable = '', remove_tables = True, frequency = 0, scan = 0, average = 0, naxis = 512, cdelt = 4.9, RL = False, clean_disk = True, calibrate = True, use_mask = True, save_model = False, niter = 100000, threshold = 40000, stokes = 'RRLL', **kwargs):
        cell = 2.45
        imsize = int(naxis * cdelt / cell)
        
        fitsTime = srh_utils.ihhmm_format(self.freqTime[frequency, scan])
        imagename = 'srh_%sT%s_%04d'%(self.hduList[0].header['DATE-OBS'].replace('-',''), fitsTime.replace(':',''), self.freqList[frequency]*1e-3 + .5)
        
        absname = os.path.join(path, imagename)
        casa_imagename = os.path.join(path, imagename)
        if calibtable!='':
            self.setFrequencyChannel(frequency)
            self.loadGains(calibtable)
        elif calibrate:
            self.calibrate(frequency)
        if use_mask:
            self.mask_name = os.path.join(path, 'srh_%sT%s_mask'%(self.hduList[0].header['DATE-OBS'], fitsTime))
            if not os.path.exists(self.mask_name):
                self.makeMask(maskname = self.mask_name)
        else:
            self.mask_name = ''
        self.saveAsUvFits(absname+'.fits', frequency=frequency, scan=scan, average=average)
        self.MSfromUvFits(absname+'.fits', absname+'.ms')
        
        if clean_disk:
            self.makeModel(modelname = casa_imagename + '_model', imagename = casa_imagename + '_temp')
            # a,b,ang = self.restoring_beam['major']['value'],self.restoring_beam['minor']['value'],self.restoring_beam['positionangle']['value']
            # rb = ['%.2farcsec'%(a*0.8), '%.2farcsec'%(b*0.8), '%.2fdeg'%ang]
            # self.clean(imagename = casa_imagename, cell = cell, imsize = imsize, niter = niter, threshold = threshold, stokes = stokes, restoringbeam=rb, usemask = 'user', mask = self.mask_name, startmodel = self.model_name, **kwargs)
            self.clean(imagename = casa_imagename, cell = cell, imsize = imsize, niter = niter, threshold = threshold, stokes = stokes, usemask = 'user', mask = self.mask_name, startmodel = self.model_name, **kwargs)
        else:
            # if clean_disk == False  -> add shift!
            
            self.clean(imagename = casa_imagename, cell = cell, imsize = imsize, niter = niter, threshold = threshold, stokes = stokes, usemask = 'user', mask = self.mask_name, **kwargs)
        self.casaImage2Fits(casa_imagename, absname, cell, imsize, naxis, scan, RL = RL, save_model = save_model, clean_disk = clean_disk)
        if remove_tables:
            rmtables(casa_imagename + '*')
            rmtables(absname + '.ms')
            os.system('rm -rf \"' + absname + '.ms.flagversions\"')
            os.system('rm \"' + absname + '.fits\"')
            
    def makeSet(self, frequencies = [0], scans = [0], start_scan = 0, step = 0, calibrate_each_scan = False, **kwargs):
        if type(frequencies) is not list and type(frequencies) is not str: frequencies = [frequencies]
        if type(scans) is not list: scans = [scans]
        
        if frequencies == 'all':
            frequencies = NP.arange(0, self.freqListLength)
            
        for f in frequencies:
            if step:
                for s in range(start_scan, self.dataLength, step):
                    if calibrate_each_scan:
                        self.makeImage(frequency = f, scan = s, calibrate = True, **kwargs)
                    else:
                        if s == start_scan:
                            self.makeImage(frequency = f, scan = s, calibrate = True, **kwargs)
                        else:
                            self.makeImage(frequency = f, scan = s, calibrate = False, **kwargs)
                    
            else:
                for s in scans:
                    if calibrate_each_scan:
                        self.makeImage(frequency = f, scan = s, calibrate = True, **kwargs)
                    else:
                        if s == scans[0]:
                            self.makeImage(frequency = f, scan = s, calibrate = True, **kwargs)
                        else:
                            self.makeImage(frequency = f, scan = s, calibrate = False, **kwargs)
             