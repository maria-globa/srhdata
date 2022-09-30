#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 18:04:32 2022

@author: maria
"""

import unittest
import os, sys
from pathlib import Path

file = Path(__file__).resolve()
parent, top = file.parent, file.parents[1]
sys.path.append(str(top))

import srhdata

class TestBasic(unittest.TestCase):
    def test_openFits(self):
        file = srhdata.open('data/srh_0612_20220502T030329.fit')
        self.assertEqual(file.isOpen, True)
        
if __name__ == '__main__':
    unittest.main()
    
from srhdata.srh_uvfits import SrhUVData
from srhdata.srh_fits_0612 import SrhFitsFile0612
from srhdata.srh_fits_0306 import SrhFitsFile0306
from casatasks import importuvfits,tclean
test_num = 17

# srh_f = SrhFitsFile0612('/home/maria/Work/SRH imaging/6-12/fits/20220502/srh_0612_20220502T030329.fit')
# srh_f.loadGains('/home/maria/Work/SRH imaging/6-12/fits/20220502/gains_ns.json')
# srh_uv = SrhUVData()
# srh_uv.write_uvfits_0612(srh_f, '/home/maria/Work/SRH imaging/srhdata_tests/test%d.fits'%test_num)

# importuvfits(fitsfile = '/home/maria/Work/SRH imaging/srhdata_tests/test%d.fits'%test_num,
#               vis = '/home/maria/Work/SRH imaging/srhdata_tests/test%d.ms'%test_num)
# tclean(vis = '/home/maria/Work/SRH imaging/srhdata_tests/test%d.ms'%test_num,
#         imagename = 'images/test%d_0'%test_num,
#         cell = 2.45,
#         imsize = 1024,
#         niter = 0,
#         stokes = 'RRLL')

# srh_f = SrhFitsFile0306('/home/maria/Work/SRH imaging/3-6/fits/20220614/srh_0306_20220614T034539.fit')
# srh_f.loadGains('/home/maria/Work/SRH imaging/3-6/fits/20220614/gains_20220614_ns.json')
# srh_uv = SrhUVData()
# srh_uv.write_uvfits_0306(srh_f, '/home/maria/Work/SRH imaging/srhdata_tests/test%d.fits'%test_num)

# importuvfits(fitsfile = '/home/maria/Work/SRH imaging/srhdata_tests/test%d.fits'%test_num,
#               vis = '/home/maria/Work/SRH imaging/srhdata_tests/test%d.ms'%test_num)
# tclean(vis = '/home/maria/Work/SRH imaging/srhdata_tests/test%d.ms'%test_num,
#         imagename = 'images/test%d_0'%test_num,
#         cell = 2.45,
#         imsize = 1024,
#         niter = 0,
#         stokes = 'RRLL')
