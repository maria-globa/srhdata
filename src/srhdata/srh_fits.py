#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 09:54:14 2022

@author: mariagloba
"""

import numpy as NP
from astropy.io import fits
from .ZirinTb import ZirinTb
import json
from .srh_coordinates import RAOcoords
from casatasks import importuvfits


class SrhFitsFile():
    def __init__(self, filenames):
        if type(filenames) is not list:
            self.filenames = [filenames]
        else:
            self.filenames = filenames
        
        self.isOpen = False;
        self.calibIndex = 0;
        self.frequencyChannel = 0;
        self.antNumberEW = 0
        self.antNumberN = 0
        self.base = 0
        self.averageCalib = False
        self.useNonlinearApproach = True
        self.obsObject = 'Sun'
        self.badAntsLcp = 0
        self.badAntsRcp = 0
        self.sizeOfUv = 1024
        self.arcsecPerPixel = 4.91104/2
        self.ZirinQSunTb = ZirinTb()
        self.convolutionNormCoef = 1
        self.useRLDif = False
        self.flux_calibrated = False
        self.corr_amp_exist = False
        
        # self.open(name)
                   
    def open(self):
        
        if self.filenames[0]:
            try:
                self.hduList = fits.open(self.filenames[0])
                self.isOpen = True
                self.dateObs = self.hduList[0].header['DATE-OBS'] + 'T' + self.hduList[0].header['TIME-OBS']
                self.antennaNumbers = self.hduList[3].data['ant_index']
                self.antennaNumbers = NP.reshape(self.antennaNumbers,self.antennaNumbers.size)
                self.antennaNames = self.hduList[2].data['ant_name']
                self.antennaNames = NP.reshape(self.antennaNames,self.antennaNames.size)
                self.antennaA = self.hduList[4].data['ant_A']
                self.antennaA = NP.reshape(self.antennaA,self.antennaA.size)
                self.antennaB = self.hduList[4].data['ant_B']
                self.antennaB = NP.reshape(self.antennaB,self.antennaB.size)
                self.antX = self.hduList[3].data['ant_X']
                self.antY = self.hduList[3].data['ant_Y']
                self.freqList = self.hduList[1].data['frequency'];
                self.freqListLength = self.freqList.size;
                self.dataLength = self.hduList[1].data['time'].size // self.freqListLength;
                self.freqTime = self.hduList[1].data['time']
                self.validScansLcp = NP.ones((self.freqListLength,self.dataLength), dtype = bool)
                self.validScansRcp = NP.ones((self.freqListLength,self.dataLength), dtype = bool)
                try:
                    self.freqTimeLcp = self.hduList[1].data['time_lcp']
                    self.freqTimeRcp = self.hduList[1].data['time_rcp']
                except:
                    pass
                self.visListLength = self.hduList[1].data['vis_lcp'].size // self.freqListLength // self.dataLength;
                self.visLcp = NP.reshape(self.hduList[1].data['vis_lcp'],(self.freqListLength,self.dataLength,self.visListLength));
                self.visRcp = NP.reshape(self.hduList[1].data['vis_rcp'],(self.freqListLength,self.dataLength,self.visListLength));
                # self.visLcp /= float(self.hduList[0].header['VIS_MAX'])
                # self.visRcp /= float(self.hduList[0].header['VIS_MAX'])
                self.ampLcp = NP.reshape(self.hduList[1].data['amp_lcp'],(self.freqListLength,self.dataLength,self.antennaNumbers.size));
                self.ampRcp = NP.reshape(self.hduList[1].data['amp_rcp'],(self.freqListLength,self.dataLength,self.antennaNumbers.size));
                ampScale = float(self.hduList[0].header['VIS_MAX']) / 128.
                self.ampLcp = self.ampLcp.astype(float) / ampScale
                self.ampRcp = self.ampRcp.astype(float) / ampScale
                try:
                    self.correctSubpacketsNumber = int(self.hduList[0].header['SUBPACKS'])
                    self.subpacketLcp = self.hduList[1].data['spacket_lcp']
                    self.subpacketRcp = self.hduList[1].data['spacket_rcp']
                    self.validScansLcp = self.subpacketLcp==self.correctSubpacketsNumber
                    self.validScansRcp = self.subpacketRcp==self.correctSubpacketsNumber
                    self.visLcp[~self.validScansLcp] = 0
                    self.visRcp[~self.validScansRcp] = 0
                    self.ampLcp[~self.validScansLcp] = 1
                    self.ampRcp[~self.validScansRcp] = 1
                    # self.calibIndex = NP.min(NP.intersect1d(NP.where(self.validScansLcp[0]), NP.where(self.validScansLcp[0]))) # frequencies?
                except:
                    pass
                try:
                    self.ampLcp_c = NP.reshape(self.hduList[1].data['amp_c_lcp'],(self.freqListLength,self.dataLength,self.antennaNumbers.size));
                    self.ampRcp_c = NP.reshape(self.hduList[1].data['amp_c_rcp'],(self.freqListLength,self.dataLength,self.antennaNumbers.size));
                    # self.ampLcp_c = self.ampLcp_c.astype(float) / ampScale
                    # self.ampRcp_c = self.ampRcp_c.astype(float) / ampScale
                    self.corr_amp_exist = True
                    self.ampLcp_c[self.ampLcp_c <= 0.01] = 1e6
                    self.ampRcp_c[self.ampRcp_c <= 0.01] = 1e6
                except:
                    pass
                
                self.RAO = RAOcoords(self.dateObs.split('T')[0], self.base*1e-3, observedObject = self.obsObject)
                self.RAO.getHourAngle(self.freqTime[0,0])
                
                self.ewAntPhaLcp = NP.zeros((self.freqListLength, self.antNumberEW))
                self.nsAntPhaLcp = NP.zeros((self.freqListLength, self.antNumberNS))
                self.ewAntPhaRcp = NP.zeros((self.freqListLength, self.antNumberEW))
                self.nsAntPhaRcp = NP.zeros((self.freqListLength, self.antNumberNS))
                self.ewLcpPhaseCorrection = NP.zeros((self.freqListLength, self.antNumberEW))
                self.ewRcpPhaseCorrection = NP.zeros((self.freqListLength, self.antNumberEW))
                self.nsLcpPhaseCorrection = NP.zeros((self.freqListLength, self.antNumberNS))
                self.nsRcpPhaseCorrection = NP.zeros((self.freqListLength, self.antNumberNS))
                self.nsLcpStair = NP.zeros(self.freqListLength)
                self.nsRcpStair = NP.zeros(self.freqListLength)
                self.nsSolarPhase = NP.zeros(self.freqListLength)
                self.ewSolarPhase = NP.zeros(self.freqListLength)
                
                self.ewAntAmpLcp = NP.ones((self.freqListLength, self.antNumberEW))
                self.nsAntAmpLcp = NP.ones((self.freqListLength, self.antNumberNS))
                self.ewAntAmpRcp = NP.ones((self.freqListLength, self.antNumberEW))
                self.nsAntAmpRcp = NP.ones((self.freqListLength, self.antNumberNS))
                
                self.ewPhaseDif = NP.zeros_like(self.ewAntPhaLcp)
                self.nsPhaseDif = NP.zeros_like(self.nsAntPhaLcp)
                self.ewAmpDif = NP.zeros_like(self.ewAntAmpLcp)
                self.nsAmpDif = NP.zeros_like(self.nsAntAmpLcp)
                
                self.nsLcpStair = NP.zeros(self.freqListLength)
                self.nsRcpStair = NP.zeros(self.freqListLength)
                self.ewSlopeLcp = NP.zeros(self.freqListLength)
                self.nsSlopeLcp = NP.zeros(self.freqListLength)
                self.ewSlopeRcp = NP.zeros(self.freqListLength)
                self.nsSlopeRcp = NP.zeros(self.freqListLength)
                self.diskLevelLcp = NP.ones(self.freqListLength)
                self.diskLevelRcp = NP.ones(self.freqListLength)
                self.lm_hd_relation = NP.ones(self.freqListLength)
                self.lcpShift = NP.zeros(self.freqListLength)
                self.rcpShift = NP.zeros(self.freqListLength)
                
                self.flags_ew = NP.array(())
                self.flags_ns = NP.array(())
                
                x_size = (self.baselines-1)*2 + self.antNumberEW + self.antNumberNS
                self.x_ini_lcp = NP.full((self.freqListLength, x_size*2+1), NP.concatenate((NP.ones(x_size+1), NP.zeros(x_size))))
                self.x_ini_rcp = NP.full((self.freqListLength, x_size*2+1), NP.concatenate((NP.ones(x_size+1), NP.zeros(x_size))))
                self.calibrationResultLcp = NP.zeros_like(self.x_ini_lcp)
                self.calibrationResultRcp = NP.zeros_like(self.x_ini_rcp)
                
                self.calibration_fun_sum_lcp = NP.zeros(self.freqListLength) # sum of residuals returned by scipy.optimize (ls_res['fun'])
                self.calibration_fun_sum_rcp = NP.zeros(self.freqListLength)
                
                self.beam_sr = NP.ones(self.freqListLength)
                
            except FileNotFoundError:
                print('File %s  not found'%self.filenames);
                
        if len(self.filenames) > 1:
            for filename in self.filenames[1:]:
                self.append(filename)
    
    def append(self,name):
        try:
            hduList = fits.open(name);
            freqTime = hduList[1].data['time']
            dataLength = hduList[1].data['time'].size // self.freqListLength;
            visLcp = NP.reshape(hduList[1].data['vis_lcp'],(self.freqListLength,dataLength,self.visListLength));
            visRcp = NP.reshape(hduList[1].data['vis_rcp'],(self.freqListLength,dataLength,self.visListLength));
            # visLcp /= float(self.hduList[0].header['VIS_MAX'])
            # visRcp /= float(self.hduList[0].header['VIS_MAX'])
            ampLcp = NP.reshape(hduList[1].data['amp_lcp'],(self.freqListLength,dataLength,self.antennaNumbers.size));
            ampRcp = NP.reshape(hduList[1].data['amp_rcp'],(self.freqListLength,dataLength,self.antennaNumbers.size));
            ampScale = float(self.hduList[0].header['VIS_MAX']) / 128.
            ampLcp = ampLcp.astype(float) / ampScale
            ampRcp = ampRcp.astype(float) / ampScale
            validScansLcp = NP.ones((self.freqListLength, dataLength), dtype = bool)
            validScansRcp = NP.ones((self.freqListLength, dataLength), dtype = bool)
            try:
                subpacketLcp = self.hduList[1].data['spacket_lcp']
                subpacketRcp = self.hduList[1].data['spacket_rcp']
                self.subpacketLcp = NP.concatenate((self.subpacketLcp, subpacketLcp), axis = 1)
                self.subpacketRcp = NP.concatenate((self.subpacketRcp, subpacketRcp), axis = 1)
                validScansLcp = subpacketLcp==self.correctSubpacketsNumber
                validScansRcp = subpacketRcp==self.correctSubpacketsNumber
                visLcp[~validScansLcp] = 0
                visRcp[~validScansRcp] = 0
                ampLcp[~validScansLcp] = 1
                ampRcp[~validScansRcp] = 1
            except:
                pass
            try:
                freqTimeLcp = hduList[1].data['time_lcp']
                freqTimeRcp = hduList[1].data['time_rcp']
                self.freqTimeLcp = NP.concatenate((self.freqTimeLcp, freqTimeLcp), axis = 1)
                self.freqTimeRcp = NP.concatenate((self.freqTimeRcp, freqTimeRcp), axis = 1)
            except:
                pass
            try:
                ampLcp_c = NP.reshape(hduList[1].data['amp_c_lcp'],(self.freqListLength,dataLength,self.antennaNumbers.size));
                ampRcp_c = NP.reshape(hduList[1].data['amp_c_rcp'],(self.freqListLength,dataLength,self.antennaNumbers.size));
                # ampLcp_c = ampLcp_c.astype(float) / ampScale
                # ampRcp_c = ampRcp_c.astype(float) / ampScale
                ampLcp_c[ampLcp_c <= 0.01] = 1e6
                ampRcp_c[ampRcp_c <= 0.01] = 1e6
                self.ampLcp_c = NP.concatenate((self.ampLcp_c, ampLcp_c), axis = 1)
                self.ampRcp_c = NP.concatenate((self.ampRcp_c, ampRcp_c), axis = 1)
            except:
                pass
            self.freqTime = NP.concatenate((self.freqTime, freqTime), axis = 1)
            self.visLcp = NP.concatenate((self.visLcp, visLcp), axis = 1)
            self.visRcp = NP.concatenate((self.visRcp, visRcp), axis = 1)
            self.ampLcp = NP.concatenate((self.ampLcp, ampLcp), axis = 1)
            self.ampRcp = NP.concatenate((self.ampRcp, ampRcp), axis = 1)
            self.validScansLcp = NP.concatenate((self.validScansLcp, validScansLcp), axis = 1)
            self.validScansRcp = NP.concatenate((self.validScansRcp, validScansRcp), axis = 1)
            self.dataLength += dataLength
            hduList.close()

        except FileNotFoundError:
            print('File %s  not found'%name);
            
    def select_scans(self, task):
        scans_in_file = self.dataLength//len(self.filenames)
        scans = NP.array((), dtype = int)
        for i in range(len(self.filenames)):
            scans = NP.append(scans, NP.array(task['task'][i]['scans']) + scans_in_file * i)
        print('selected scans: ', scans)
        self.dataLength = len(scans)
        self.freqTime = self.freqTime[:, scans]
        try:
            self.freqTimeLcp = self.freqTimeLcp[:, scans]
            self.freqTimeRcp = self.freqTimeRcp[:, scans]
        except:
            pass
        self.visLcp = self.visLcp[:, scans, :]
        self.visRcp = self.visRcp[:, scans, :]
        self.ampLcp = self.ampLcp[:, scans, :]
        self.ampRcp = self.ampRcp[:, scans, :]
        try:
            self.ampLcp_c = self.ampLcp_c[:, scans, :]
            self.ampRcp_c = self.ampRcp_c[:, scans, :]
        except:
            pass
        try:
            self.subpacketLcp = self.subpacketLcp[:, scans]
            self.subpacketRcp = self.subpacketRcp[:, scans]
            self.validScansLcp = self.validScansLcp[:, scans]
            self.validScansRcp = self.validScansRcp[:, scans]
        except:
            pass
            
    def calibrate(self, freq = 'all', phaseCorrect = True, amplitudeCorrect = True, average = 20):
        if freq == 'all':
            self.calculatePhaseCalibration()
            for freq in range(self.freqListLength):
                self.setFrequencyChannel(freq)
                self.vis2uv(scan = self.calibIndex, phaseCorrect=phaseCorrect, amplitudeCorrect=amplitudeCorrect, average=average)
                self.centerDisk()
        else:
            self.setFrequencyChannel(freq)
            self.solarPhase(freq)
            self.updateAntennaPhase(freq)
            self.vis2uv(scan = self.calibIndex, phaseCorrect=phaseCorrect, amplitudeCorrect=amplitudeCorrect, average=average)
            self.centerDisk()
            
    def image(self, freq = 0, scan = 0, average = 0, polarization = 'both', phaseCorrect = True, amplitudeCorrect = True, frame = 'heliocentric'):
        self.setFrequencyChannel(freq)    
        self.vis2uv(scan = scan, phaseCorrect=phaseCorrect, amplitudeCorrect=amplitudeCorrect, average=average)
        self.uv2lmImage()
        if frame == 'heliocentric':
            self.lm2Heliocentric()
        if polarization == 'both':
            return NP.flip(self.lcp, 0), NP.flip(self.rcp, 0)
        elif polarization == 'lcp':
            return NP.flip(self.lcp, 0)
        elif polarization == 'rcp':
            return NP.flip(self.rcp, 0)
        
    def saveGains(self, filename):
        currentGainsDict = {}
        currentGainsDict['ewPhaseLcp'] = (self.ewAntPhaLcp + self.ewLcpPhaseCorrection).tolist()
        currentGainsDict['nsPhaseLcp'] = (self.nsAntPhaLcp + self.nsLcpPhaseCorrection).tolist()
        currentGainsDict['ewPhaseRcp'] = (self.ewAntPhaRcp + self.ewRcpPhaseCorrection).tolist()
        currentGainsDict['nsPhaseRcp'] = (self.nsAntPhaRcp + self.nsRcpPhaseCorrection).tolist()
        currentGainsDict['ewAmpLcp'] = self.ewAntAmpLcp.tolist()
        currentGainsDict['nsAmpLcp'] = self.nsAntAmpLcp.tolist()
        currentGainsDict['ewAmpRcp'] = self.ewAntAmpRcp.tolist()
        currentGainsDict['nsAmpRcp'] = self.nsAntAmpRcp.tolist()
        currentGainsDict['rcpShift'] = self.rcpShift.tolist()
        currentGainsDict['lcpShift'] = self.lcpShift.tolist()
        currentGainsDict['lm_hd_relation'] = self.lm_hd_relation.tolist()
        with open(filename, 'w') as saveGainFile:
            json.dump(currentGainsDict, saveGainFile)
            
    def loadGains(self, gains):
        if type(gains) == 'string':
            self.loadGainsFromFile(gains)
        if type(gains) == dict:
            self.loadGainsFromDict(gains)
            
    def loadGainsFromDict(self, gains):
        if self.hduList[0].header['INSTRUME'] == gains['array']:
            try:
                frequency = NP.where(self.freqList==gains['frequency']*1e3)[0][0]
            except:
                raise Exception('Frequency %d is not in the list' % gains['frequency'])

            self.nsAntAmpRcp[frequency] = NP.array(gains['gains_R_amplitude'][:self.antNumberNS])
            self.ewAntAmpRcp[frequency] = NP.array(gains['gains_R_amplitude'][self.antNumberNS:])
            self.nsAntAmpLcp[frequency] = NP.array(gains['gains_L_amplitude'][:self.antNumberNS])
            self.ewAntAmpLcp[frequency] = NP.array(gains['gains_L_amplitude'][self.antNumberNS:])
            self.nsAntPhaRcp[frequency] = NP.array(gains['gains_R_phase'][:self.antNumberNS])
            self.ewAntPhaRcp[frequency] = NP.array(gains['gains_R_phase'][self.antNumberNS:])
            self.nsAntPhaLcp[frequency] = NP.array(gains['gains_L_phase'][:self.antNumberNS])
            self.ewAntPhaLcp[frequency] = NP.array(gains['gains_L_phase'][self.antNumberNS:])
            self.calibration_fun_sum_rcp[frequency] = gains['residual_R']
            self.calibration_fun_sum_lcp[frequency] = gains['residual_L']
        else:
            raise Exception('Attempted to load %s gains for %s array' % (gains['array'], self.hduList[0].header['INSTRUME']))
            
    def loadGainsFromFile(self, filename):
        with open(filename,'r') as readGainFile:
            currentGains = json.load(readGainFile)
        self.ewAntPhaLcp = NP.array(currentGains['ewPhaseLcp'])
        self.ewAntPhaRcp = NP.array(currentGains['ewPhaseRcp'])
        self.ewAntAmpLcp = NP.array(currentGains['ewAmpLcp'])
        self.ewAntAmpRcp = NP.array(currentGains['ewAmpRcp'])
        try:
            self.nsAntPhaLcp = NP.array(currentGains['nsPhaseLcp'])
            self.nsAntPhaRcp = NP.array(currentGains['nsPhaseRcp'])
            self.nsAntAmpLcp = NP.array(currentGains['nsAmpLcp'])
            self.nsAntAmpRcp = NP.array(currentGains['nsAmpRcp'])
        except:
            pass
        try:
            self.nsAntPhaLcp = NP.array(currentGains['nPhaseLcp'])
            self.nsAntPhaRcp = NP.array(currentGains['nPhaseRcp'])
            self.nsAntAmpLcp = NP.array(currentGains['nAmpLcp'])
            self.nsAntAmpRcp = NP.array(currentGains['nAmpRcp'])
        except:
            pass
        try:
            self.nsAntPhaLcp = NP.array(currentGains['sPhaseLcp'])
            self.nsAntPhaRcp = NP.array(currentGains['sPhaseRcp'])
            self.nsAntAmpLcp = NP.array(currentGains['sAmpLcp'])
            self.nsAntAmpRcp = NP.array(currentGains['sAmpRcp'])
        except:
            pass
        try:
            self.rcpShift = NP.array(currentGains['rcpShift'])
            self.lcpShift = NP.array(currentGains['lcpShift'])
        except:
            pass
        self.lm_hd_relation = NP.array(currentGains['lm_hd_relation'])
        
    def getGains(self, frequency):
        ant_mask = NP.ones(len(self.antennaNames), dtype = int)
        ant_mask[NP.append(self.flags_ns, self.flags_ew+self.antNumberNS)] = 0
        gains_dict = {}
        gains_dict['time'] = self.dateObs
        gains_dict['array'] = self.hduList[0].header['INSTRUME']
        gains_dict['algorithm'] = 'globa'
        gains_dict['frequency'] = int(self.freqList[frequency]*1e-3)
        gains_dict['antennas'] = self.antennaNames.tolist()
        gains_dict['gains_R_amplitude'] = NP.append(self.nsAntAmpRcp[frequency], self.ewAntAmpRcp[frequency]).tolist()
        gains_dict['gains_L_amplitude'] = NP.append(self.nsAntAmpLcp[frequency], self.ewAntAmpLcp[frequency]).tolist()
        gains_dict['gains_R_phase'] = NP.append(self.nsAntPhaRcp[frequency], self.ewAntPhaRcp[frequency]).tolist()
        gains_dict['gains_L_phase'] = NP.append(self.nsAntPhaLcp[frequency], self.ewAntPhaLcp[frequency]).tolist()
        gains_dict['residual_R'] = float(self.calibration_fun_sum_rcp[frequency])
        gains_dict['residual_L'] = float(self.calibration_fun_sum_lcp[frequency])
        gains_dict['additional'] = {}
        gains_dict['antenna_mask'] = ant_mask.tolist()
        return gains_dict
        
    def loadRLdif(self, filename):
        with open(filename,'r') as readRLDifFile:
            RLDif = json.load(readRLDifFile)
        self.ewPhaseDif = NP.array(RLDif['ewPhaseDif'])
        self.nsPhaseDif = NP.array(RLDif['nsPhaseDif'])
        self.ewAmpDif = NP.array(RLDif['ewAmpDif'])
        self.nsAmpDif = NP.array(RLDif['nsAmpDif'])

    def changeObject(self, obj):
        self.obsObject = obj
        self.RAO = RAOcoords(self.dateObs.split('T')[0], self.base, observedObject = self.obsObject)

    def setSizeOfUv(self, sizeOfUv):
        self.sizeOfUv = sizeOfUv
        self.uvLcp = NP.zeros((self.sizeOfUv,self.sizeOfUv),dtype=complex);
        self.uvRcp = NP.zeros((self.sizeOfUv,self.sizeOfUv),dtype=complex);
        
    def close(self):
        self.hduList.close();
        
    def setBaselinesNumber(self, value):
        self.baselines = value
        x_size = (self.baselines-1)*2 + self.antNumberEW + self.antNumberNS
        self.x_ini_lcp = NP.full((self.freqListLength, x_size*2+1), NP.concatenate((NP.ones(x_size+1), NP.zeros(x_size))))
        self.x_ini_rcp = NP.full((self.freqListLength, x_size*2+1), NP.concatenate((NP.ones(x_size+1), NP.zeros(x_size))))
        self.calibrationResultLcp = NP.zeros_like(self.x_ini_lcp)
        self.calibrationResultRcp = NP.zeros_like(self.x_ini_rcp)
    
    def calculatePhaseCalibration(self, lcp = True, rcp = True):
       for freq in range(self.freqListLength):
           self.solarPhase(freq)
           self.updateAntennaPhase(freq, lcp = lcp, rcp = rcp)

    def updateAntennaPhase(self, freqChannel, lcp = True, rcp = True):
        if self.useNonlinearApproach:
            if lcp:
                self.calculatePhaseLcp_nonlinear(freqChannel)
            if rcp:
                self.calculatePhaseRcp_nonlinear(freqChannel)
            if rcp and lcp:
                flags_ew_lcp = NP.where(self.ewAntAmpLcp[freqChannel] == 1e6)[0]
                flags_ew_rcp = NP.where(self.ewAntAmpRcp[freqChannel] == 1e6)[0]
                self.flags_ew = NP.unique(NP.append(flags_ew_lcp, flags_ew_rcp))
                flags_ns_lcp = NP.where(self.nsAntAmpLcp[freqChannel] == 1e6)[0]
                flags_ns_rcp = NP.where(self.nsAntAmpRcp[freqChannel] == 1e6)[0]
                self.flags_ns = NP.unique(NP.append(flags_ns_lcp, flags_ns_rcp))
                self.ewAntAmpLcp[freqChannel][self.flags_ew] = 1e6
                self.nsAntAmpLcp[freqChannel][self.flags_ns] = 1e6
                self.ewAntAmpRcp[freqChannel][self.flags_ew] = 1e6
                self.nsAntAmpRcp[freqChannel][self.flags_ns] = 1e6
                
    def MSfromUvFits(self, uvfits_name, ms_name):
        self.uvfits_name = uvfits_name
        self.ms_name = ms_name
        importuvfits(uvfits_name, ms_name)

    def solarPhase(self, freq):
        pass

    def calculatePhaseLcp_nonlinear(self, freqChannel):
        pass
        
    def calculatePhaseRcp_nonlinear(self, freqChannel):
        pass

    def allGainsFunc_constrained(self, x, obsVis, freq):
        pass 
    
    def buildEwPhase(self):
        pass
        
    def buildNSPhase(self):
        pass
    
    def setCalibIndex(self, calibIndex):
        self.calibIndex = calibIndex;

    def setFrequencyChannel(self, channel):
        self.frequencyChannel = channel
        
    def vis2uv(self, scan, phaseCorrect = True, amplitudeCorrect = False, PSF=False, average = 0):
        pass

    def uv2lmImage(self):
        pass
        
    def lm2Heliocentric(self):
        pass

    def createDisk(self, radius, arcsecPerPixel = 2.45552):
        pass
    
    def createUvUniform(self):
        pass
                    
    def createUvPsf(self, T, ewSlope, nSlope, shift):
        pass
    
    def diskDiff(self, x, pol):
        pass
    
    def findDisk(self):
        pass
        
    def diskDiff_2(self, x, pol):
        pass

    def centerDisk(self):
        pass
        
    def modelDiskConv(self):
        pass
        
    def modelDiskConv_unity(self):
        pass
    
    def saveAsUvFits(self):
        pass
    
    def clean(self):
        pass