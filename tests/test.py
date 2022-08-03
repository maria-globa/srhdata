#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 18:04:32 2022

@author: maria
"""

import unittest
import srhdata


class TestBasic(unittest.TestCase):
    def test_openFits(self):
        file = srhdata.open('/home/maria/Work/SRH imaging/6-12/fits/20220519/srh_0612_20220519T033044.fit')
        self.assertEqual(file.isOpen, True)
        
if __name__ == '__main__':
    unittest.main()