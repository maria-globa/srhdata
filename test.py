#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 18:04:32 2022

@author: maria
"""

import unittest
from src import srhdata


class TestBasic(unittest.TestCase):
    def test_openFits(self):
        file = srhdata.open('data/srh_0612_20220502T030329.fit')
        self.assertEqual(file.isOpen, True)
        
if __name__ == '__main__':
    unittest.main()