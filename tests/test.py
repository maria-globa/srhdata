#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 18:04:32 2022

@author: maria
"""

import unittest
import os, sys
from pathlib import Path

import srhdata

import ftplib
ftp = ftplib.FTP('ftp.rao.istp.ac.ru', 'anonymous', 'anonymous')
ftp.cwd('SRH/SRH0612/2022/01/13')
filename = 'srh_0612_20220113T031226.fit'
with open( filename, 'wb' ) as file :
    ftp.retrbinary('RETR %s' % filename, file.write)
    file.close()

class TestBasic(unittest.TestCase):
    def test_openFits(self):
        file = srhdata.open('srh_0612_20220113T031226.fit')
        self.assertEqual(file.isOpen, True)
        
if __name__ == '__main__':
    unittest.main()
    
# from srhdata.srh_uvfits import SrhUVData
# from srhdata.srh_fits_0612 import SrhFitsFile0612
# from srhdata.srh_fits_0306 import SrhFitsFile0306
# from casatasks import importuvfits,tclean
# test_num = 16

# # srh_f = SrhFitsFile0612('/home/mariagloba/Work/Python Scripts/6-12/fits/20220515/srh_0612_20220515T033902.fit')
# # srh_f.makeImage(path = '/home/mariagloba/Work/Python Scripts/srhdata_tests', cleantables = False, scan = 2, average = 10, frequency = 0, deconvolver = 'multiscale', scales = [1,2,3,4,5,10,15,30])


# srh_f = SrhFitsFile0306('/home/mariagloba/Work/fits/20220614/srh_0306_20220614T034539.fit')
# srh_f.makeImage(path = '/home/mariagloba/Work/Python Scripts/srhdata_tests', remove_tables = False, clean_disk = False, scan = 5, average = 10, frequency = 3)#, deconvolver = 'multiscale', scales = [1,2,3,4,5,10,15,30])

# srh_f = SrhFitsFile0306('/home/mariagloba/Work/fits/2021/04/30/srh_20210430T030553.fit')
# srh_f.makeImage(path = '/home/mariagloba/Work/Python Scripts/srhdata_tests', remove_tables = True, clean_disk = False, scan = 0, average = 20, frequency = 0)#, deconvolver = 'multiscale', scales = [1,2,3,4,5,10,15,30])

