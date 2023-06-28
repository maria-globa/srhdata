#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 11:44:53 2022

@author: sergey_lesovoi
"""

from pyuvdata import UVData
import numpy as NP
from astropy.time import Time
import astropy.coordinates as COORD
from astropy.coordinates import EarthLocation
from astropy import units as u
from datetime import timedelta
from .srh_coordinates import base2uvw0306, base2uvw0612

class SrhUVData(UVData):
    """
    """
    def __init__(self):
        self.srhLat = 51.759
        self.srhLon = 102.217
        self.srhAlt = 799.
        self.srhEarthLocation = EarthLocation(lon=self.srhLon * u.deg, lat=self.srhLat * u.deg, height=self.srhAlt * u.m)
        self.srhLocation = self.srhEarthLocation.T.to_value()
        self.srhLat = NP.deg2rad(self.srhLat)
        self.srhLon = NP.deg2rad(self.srhLon)
        self.phi_operator = NP.array([
            [-NP.sin(self.srhLat), 0., NP.cos(self.srhLat)],
            [0., 1., 0.],
            [NP.cos(self.srhLat), 0., NP.sin(self.srhLat)]
            ])
        super().__init__()
        
    def write_uvfits_0306(self, srhFits, dstName, frequency=0, scan=0, average=0):
        lastVisibility=3007
        antZeroRow = srhFits.antZeroRow[:32]
        vis_list = NP.append(NP.arange(0, lastVisibility), antZeroRow).astype(int)
        
        if scan < 0 or scan > srhFits.dataLength:
            raise Exception('scan is out of range')
        if scan + average > srhFits.dataLength:
            raise Exception('averaging range is larger than data range')
        if average > 1:
            visRcp = NP.mean(srhFits.visRcp[frequency, scan:scan+average, vis_list], 1)
            visLcp = NP.mean(srhFits.visLcp[frequency, scan:scan+average, vis_list], 1)
        else:
            visRcp = srhFits.visRcp[frequency, scan, vis_list].copy()
            visLcp = srhFits.visLcp[frequency, scan, vis_list].copy()

            
        visRcp[lastVisibility:] = NP.conj(visRcp[lastVisibility:])
        visLcp[lastVisibility:] = NP.conj(visLcp[lastVisibility:])

        scanTime = Time(srhFits.dateObs.split('T')[0] + 'T' + str(timedelta(seconds=srhFits.freqTime[frequency,scan])))
        coords = COORD.get_sun(scanTime)
        
        for vis in range(lastVisibility):
            i = vis // 97
            j = vis % 97 
            visLcp[vis] *= NP.exp(1j*(-(srhFits.ewAntPhaLcp[frequency, j]+srhFits.ewLcpPhaseCorrection[frequency, j]) + (srhFits.nsAntPhaLcp[frequency, i] + srhFits.nsLcpPhaseCorrection[frequency, i])))
            visRcp[vis] *= NP.exp(1j*(-(srhFits.ewAntPhaRcp[frequency, j]+srhFits.ewRcpPhaseCorrection[frequency, j]) + (srhFits.nsAntPhaRcp[frequency, i] + srhFits.nsRcpPhaseCorrection[frequency, i])))
            visLcp[vis] /= (srhFits.ewAntAmpLcp[frequency, j] * srhFits.nsAntAmpLcp[frequency, i])
            visRcp[vis] /= (srhFits.ewAntAmpRcp[frequency, j] * srhFits.nsAntAmpRcp[frequency, i])
        
        for vis_zr in range(len(antZeroRow)):
            vis = lastVisibility + vis_zr
            visLcp[vis] *= NP.exp(1j*((srhFits.ewAntPhaLcp[frequency, vis_zr]+srhFits.ewLcpPhaseCorrection[frequency, vis_zr]) - (srhFits.ewAntPhaLcp[frequency, 32] + srhFits.ewLcpPhaseCorrection[frequency, 32])))
            visRcp[vis] *= NP.exp(1j*((srhFits.ewAntPhaRcp[frequency, vis_zr]+srhFits.ewRcpPhaseCorrection[frequency, vis_zr]) - (srhFits.ewAntPhaRcp[frequency, 32] + srhFits.ewRcpPhaseCorrection[frequency, 32])))
            visLcp[vis] /= (srhFits.ewAntAmpLcp[frequency, vis_zr] * srhFits.ewAntAmpLcp[frequency, 32])
            visRcp[vis] /= (srhFits.ewAntAmpRcp[frequency, vis_zr] * srhFits.ewAntAmpRcp[frequency, 32])

        self.ant_1_array = srhFits.antennaA[vis_list]
        self.ant_2_array = srhFits.antennaB[vis_list]
        self.antenna_names = srhFits.antennaNames
        self.antenna_numbers = NP.array(list(map(int,srhFits.antennaNumbers)))
        self.Nants_data = NP.union1d(srhFits.antennaA[vis_list],srhFits.antennaB[vis_list]).size
        self.Nants_telescope = srhFits.antennaNames.shape[0]
        self.Nfreqs = 1
        self.Npols = 2
        self.Ntimes = 1
        self.Nspws = 1
        self.Nbls = len(vis_list)
        self.Nblts = self.Nbls * self.Ntimes
        self.phase_type = 'phased'
        self.phase_center_ra = coords.ra.rad
        self.phase_center_dec = coords.dec.rad
        self.phase_center_app_ra = NP.full(self.Nblts,coords.ra.rad)
        self.phase_center_app_dec = NP.full(self.Nblts,coords.dec.rad)
        self.phase_center_epoch = 2000.0
        self.channel_width = 1e7
        self.freq_array = NP.zeros((1,1))
        self.freq_array[0] = srhFits.freqList[frequency]*1e3
        self.history = 'SRH'
        self.instrument = 'SRH0306'
        self.integration_time = NP.full(self.Nblts,0.1)
        self.antenna_diameters = NP.full(self.Nants_telescope,3.)
        self.lst_array = NP.full(self.Nblts,0.)
        self.object_name = 'Sun'
        self.polarization_array = NP.array([-1,-2])
        self.spw_array = [1]
        self.telescope_location = list(self.srhLocation)
        self.telescope_name = 'SRH'
        self.time_array = NP.full(self.Nblts,scanTime.jd)
        self.data_array = NP.zeros((self.Nblts,1,self.Nfreqs,self.Npols),dtype='complex')
        self.data_array[:,0,0,0] = visRcp
        self.data_array[:,0,0,1] = visLcp
        
        self.flag_array = NP.full((self.Nblts,1,self.Nfreqs,self.Npols),False,dtype='bool')
        self.nsample_array = NP.full((self.Nblts,1,self.Nfreqs,self.Npols),1,dtype='float')
        self.vis_units = 'uncalib'
        
        flags_ew_lcp = NP.where(srhFits.ewAntAmpLcp[frequency] == 1e6)[0]
        flags_ew_rcp = NP.where(srhFits.ewAntAmpRcp[frequency] == 1e6)[0]
        flags_ew = NP.unique(NP.append(flags_ew_lcp, flags_ew_rcp))
        flags_ns_lcp = NP.where(srhFits.nsAntAmpLcp[frequency] == 1e6)[0]
        flags_ns_rcp = NP.where(srhFits.nsAntAmpRcp[frequency] == 1e6)[0]
        flags_ns = NP.unique(NP.append(flags_ns_lcp, flags_ns_rcp))
        
        flags_arr = NP.zeros((31,97), dtype = 'bool')
        flags_arr[flags_ns,:] = True
        flags_arr[:,flags_ew] = True
        flags_arr = NP.reshape(flags_arr, (31*97))
        flags_zr = NP.zeros(len(antZeroRow), dtype = 'bool')
        flags_zr[flags_ew[flags_ew < 32]] = True
        flags_arr = NP.append(flags_arr, flags_zr)
        
        self.flag_array[:,0,0,0] = flags_arr
        self.flag_array[:,0,0,1] = flags_arr

        self.antenna_positions = NP.zeros((self.Nants_telescope,3))
        for ant in NP.arange(0, 97):
            self.antenna_positions[ant] = [0, (ant - 32) * 9.8, 0]
        for ant in NP.arange(97, 128):
            self.antenna_positions[ant] =  [-(ant - 96) * 9.8, 0, 0]
        self.baseline_array = 2048 * (self.ant_2_array + 1) + self.ant_1_array + 1 + 2**16
#        self.uvw_array = self.antenna_positions[self.ant_1_array] - self.antenna_positions[self.ant_2_array]
        self.uvw_array = NP.zeros((len(vis_list),3))

        lst = Time(scanTime,scale='utc', location=self.srhEarthLocation).sidereal_time('mean')
        hourAngle = lst.to('rad').value - self.phase_center_ra
        for vis in range(len(vis_list)):
            self.uvw_array[vis] = base2uvw0306(hourAngle,coords.dec.rad,self.ant_2_array[vis], self.ant_1_array[vis])
        
        super().write_uvfits(dstName,write_lst=False,spoof_nonessential=True,run_check=False)

    def write_uvfits_0612(self, srhFits, dstName, frequency=0, scan=0, average=0):
        lastVisibility=8192
        if scan < 0 or scan > srhFits.dataLength:
            raise Exception('scan is out of range')
        if scan + average > srhFits.dataLength:
            raise Exception('averaging range is larger than data range')
        if average > 1:
            visRcp = NP.mean(srhFits.visRcp[frequency, scan:scan+average, 0:lastVisibility], 0)
            visLcp = NP.mean(srhFits.visLcp[frequency, scan:scan+average, 0:lastVisibility], 0)
        else:
            visRcp = srhFits.visRcp[frequency, scan, 0:lastVisibility].copy()
            visLcp = srhFits.visLcp[frequency, scan, 0:lastVisibility].copy()
        
        scanTime = Time(srhFits.dateObs.split('T')[0] + 'T' + str(timedelta(seconds=srhFits.freqTime[frequency,scan])))
        coords = COORD.get_sun(scanTime)
        
        for vis in range(lastVisibility):
            i = vis // 128
            j = vis % 128 
            visLcp[vis] *= NP.exp(1j*(-(srhFits.ewAntPhaLcp[frequency, j]+srhFits.ewLcpPhaseCorrection[frequency, j]) + (srhFits.nsAntPhaLcp[frequency, i] + srhFits.nsLcpPhaseCorrection[frequency, i])))
            visRcp[vis] *= NP.exp(1j*(-(srhFits.ewAntPhaRcp[frequency, j]+srhFits.ewRcpPhaseCorrection[frequency, j]) + (srhFits.nsAntPhaRcp[frequency, i] + srhFits.nsRcpPhaseCorrection[frequency, i])))
            visLcp[vis] /= (srhFits.ewAntAmpLcp[frequency, j] * srhFits.nsAntAmpLcp[frequency, i])
            visRcp[vis] /= (srhFits.ewAntAmpRcp[frequency, j] * srhFits.nsAntAmpRcp[frequency, i])
        
        self.ant_1_array = srhFits.antennaA[0:lastVisibility]
        self.ant_2_array = srhFits.antennaB[0:lastVisibility]
        self.antenna_names = srhFits.antennaNames
        self.antenna_numbers = NP.array(list(map(int,srhFits.antennaNumbers)))
        self.Nants_data = NP.union1d(srhFits.antennaA[0:lastVisibility],srhFits.antennaB[0:lastVisibility]).size
        self.Nants_telescope = srhFits.antennaNames.shape[0]
        self.Nfreqs = 1
        self.Npols = 2
        self.Ntimes = 1
        self.Nspws = 1
        self.Nbls = lastVisibility
        self.Nblts = lastVisibility * self.Ntimes
        self.phase_type = 'phased'
        self.phase_center_ra = coords.ra.rad
        self.phase_center_dec = coords.dec.rad
        self.phase_center_app_ra = NP.full(self.Nblts,coords.ra.rad)
        self.phase_center_app_dec = NP.full(self.Nblts,coords.dec.rad)
        self.phase_center_epoch = 2000.0
        self.channel_width = 1e7
        self.freq_array = NP.zeros((1,1))
        self.freq_array[0] = srhFits.freqList[frequency]*1e3
        self.history = 'SRH'
        self.instrument = 'SRH'
        self.integration_time = NP.full(self.Nblts,0.1)
        self.antenna_diameters = NP.full(self.Nants_telescope,2.)
        self.lst_array = NP.full(self.Nblts,0.)
        self.object_name = 'Sun'
        self.polarization_array = NP.array([-1,-2])
        self.spw_array = [1]
        self.telescope_location = list(self.srhLocation)
        self.telescope_name = 'SRH'
        self.time_array = NP.full(self.Nblts,scanTime.jd)
        self.data_array = NP.zeros((self.Nblts,1,self.Nfreqs,self.Npols),dtype='complex')
        self.data_array[:,0,0,0] = visRcp
        self.data_array[:,0,0,1] = visLcp
            
        self.flag_array = NP.full((self.Nblts,1,self.Nfreqs,self.Npols),False,dtype='bool')
        self.nsample_array = NP.full((self.Nblts,1,self.Nfreqs,self.Npols),1,dtype='float')
        self.vis_units = 'uncalib'

        flags_ew_lcp = NP.where(srhFits.ewAntAmpLcp[frequency] == 1e6)[0]
        flags_ew_rcp = NP.where(srhFits.ewAntAmpRcp[frequency] == 1e6)[0]
        flags_ew = NP.unique(NP.append(flags_ew_lcp, flags_ew_rcp))
        flags_ns_lcp = NP.where(srhFits.nsAntAmpLcp[frequency] == 1e6)[0]
        flags_ns_rcp = NP.where(srhFits.nsAntAmpRcp[frequency] == 1e6)[0]
        flags_ns = NP.unique(NP.append(flags_ns_lcp, flags_ns_rcp))
        
        flags_arr = NP.zeros((64,128), dtype = 'bool')
        flags_arr[flags_ns,:] = True
        flags_arr[:,flags_ew] = True
        flags_arr = NP.reshape(flags_arr, (64*128))
        
        self.flag_array[:,0,0,0] = flags_arr
        self.flag_array[:,0,0,1] = flags_arr

        self.antenna_positions = NP.zeros((self.Nants_telescope,3))
        for ant in NP.arange(0, 128):
            self.antenna_positions[ant] = [0, (ant - 63.5) * 4.9, 0]
        for ant in NP.arange(128, 192):
            self.antenna_positions[ant] =  [(ant - 127.5) * 4.9, 0, 0]
        self.baseline_array = 2048 * (self.ant_2_array + 1) + self.ant_1_array + 1 + 2**16
        self.uvw_array = self.antenna_positions[self.ant_1_array] - self.antenna_positions[self.ant_2_array]

        lst = Time(scanTime,scale='utc', location=self.srhEarthLocation).sidereal_time('mean')
        hourAngle = lst.to('rad').value - self.phase_center_ra
        for vis in range(lastVisibility):
            self.uvw_array[vis] = base2uvw0612(hourAngle,coords.dec.rad,self.ant_2_array[vis] + 1, self.ant_1_array[vis] + 1)
        
        super().write_uvfits(dstName,write_lst=False,spoof_nonessential=True,run_check=False)
