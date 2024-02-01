#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 10:06:59 2022

@author: mariagloba
"""
import numpy as NP
import ephem
from astropy import constants
import sunpy.coordinates
from astropy import coordinates

def base2uvw0306(hourAngle, declination, antenna0, antenna1):
    phi = 0.903338787600965
    if ((antenna0 >= 1 and antenna0 <= 97) and (antenna1 >= 98 and antenna1 <= 129)):
        base = NP.array([-antenna1 + 97, antenna0 - 33, 0.])
    elif ((antenna1 >= 1 and antenna1 <= 97) and (antenna0 >= 98 and antenna0 <= 129)):
        base = NP.array([-antenna0 + 97, antenna1 - 33, 0.])
    elif ((antenna0 >= 1 and antenna0 <= 97) and (antenna1 >= 1 and antenna1 <= 97)):
        base = NP.array([0.,antenna0 - antenna1,0.])
    elif ((antenna0 >= 98 and antenna0 <= 129) and (antenna1 >= 98 and antenna1 <= 129)):
        base = NP.array([-antenna0 + antenna1,0.,0.])
    
    base *= 9.8;
    
    phi_operator = NP.array([
        [-NP.sin(phi), 0., NP.cos(phi)],
        [0., 1., 0.],
        [NP.cos(phi), 0., NP.sin(phi)]
        ])

    uvw_operator = NP.array([
        [ NP.sin(hourAngle),		 NP.cos(hourAngle),		0.	  ],
        [-NP.sin(declination)*NP.cos(hourAngle),  NP.sin(declination)*NP.sin(hourAngle), NP.cos(declination)], 
        [ NP.cos(declination)*NP.cos(hourAngle), -NP.cos(declination)*NP.sin(hourAngle), NP.sin(declination)]  
        ])

    return NP.dot(uvw_operator, NP.dot(phi_operator, base))

def base2uvw0612(hourAngle, declination, antenna0, antenna1):
    phi = 0.903338787600965
    if ((antenna0 >= 1 and antenna0 <= 128) and (antenna1 >= 129 and antenna1 <= 192)):
        base = NP.array([antenna1 - 128.5,antenna0 - 64.5,0.])
    elif ((antenna1 >= 1 and antenna1 <= 128) and (antenna0 >= 129 and antenna0 <= 192)):
        base = NP.array([antenna0 - 128.5,antenna1 - 64.5,0.])
    elif ((antenna0 >= 1 and antenna0 <= 128) and (antenna1 >= 1 and antenna1 <= 128)):
        base = NP.array([0.,antenna0 - antenna1,0.])
    elif ((antenna0 >= 129 and antenna0 <= 192) and (antenna1 >= 129 and antenna1 <= 192)):
        base = NP.array([antenna0 - antenna1,0.,0.])
    
    base *= 4.9;
    
    phi_operator = NP.array([
        [-NP.sin(phi), 0., NP.cos(phi)],
        [0., 1., 0.],
        [NP.cos(phi), 0., NP.sin(phi)]
        ])

    uvw_operator = NP.array([
        [ NP.sin(hourAngle),		 NP.cos(hourAngle),		0.	  ],
        [-NP.sin(declination)*NP.cos(hourAngle),  NP.sin(declination)*NP.sin(hourAngle), NP.cos(declination)], 
        [ NP.cos(declination)*NP.cos(hourAngle), -NP.cos(declination)*NP.sin(hourAngle), NP.sin(declination)]  
        ])

    return NP.dot(uvw_operator, NP.dot(phi_operator, base))


def distFromCenter(ant):
    if ant < 140:
        n = NP.abs(ant - 70)
        if n<25:
            baseDist = n
        else:
            baseDist = 26
            baseDist += (n-24)*2
            if n>48:
                baseDist += (n-48)*2
        return baseDist * NP.sign(ant-70)
    else:
        n = ant - 139
        if n<25:
            baseDist = n
        else:
            baseDist = 24
            baseDist += (n-24)*2
            if n>48:
                baseDist += (n-48)*2
        return baseDist
    

def base2uvw1224(hourAngle, declination, antenna0, antenna1):
    phi = 0.903338787600965
    if ((antenna0 >= 0 and antenna0 <= 138) and (antenna1 >= 139 and antenna1 <= 207)):
        base = NP.array([distFromCenter(antenna1), distFromCenter(antenna0), 0.])
        
    elif ((antenna1 >= 0 and antenna1 <= 138) and (antenna0 >= 139 and antenna0 <= 207)):
        base = NP.array([distFromCenter(antenna0), distFromCenter(antenna1), 0.])
        
    elif ((antenna0 >= 0 and antenna0 <= 138) and (antenna1 >= 0 and antenna1 <= 138)):
        base = NP.array([0.,distFromCenter(antenna0)-distFromCenter(antenna1),0.])
        
    elif ((antenna0 >= 139 and antenna0 <= 207) and (antenna1 >= 139 and antenna1 <= 207)):
        base = NP.array([distFromCenter(antenna0)-distFromCenter(antenna1),0.,0.])
    
    base *= 2.45;
    
    phi_operator = NP.array([
        [-NP.sin(phi), 0., NP.cos(phi)],
        [0., 1., 0.],
        [NP.cos(phi), 0., NP.sin(phi)]
        ])

    uvw_operator = NP.array([
        [ NP.sin(hourAngle),		 NP.cos(hourAngle),		0.	  ],
        [-NP.sin(declination)*NP.cos(hourAngle),  NP.sin(declination)*NP.sin(hourAngle), NP.cos(declination)], 
        [ NP.cos(declination)*NP.cos(hourAngle), -NP.cos(declination)*NP.sin(hourAngle), NP.sin(declination)]  
        ])

    return NP.dot(uvw_operator, NP.dot(phi_operator, base))

class RAOcoords():
    def __init__(self, theDate, base, observedObject = 'Sun'):
        self.base = base
        self.observatory = ephem.Observer()
        self.observatory.lon = NP.deg2rad(102.217)
        self.observatory.lat = NP.deg2rad(51.759)
        self.observatory.elev= 799
        self.observatory.date= theDate
        if observedObject == 'Moon' or observedObject == 'moon':
            self.obsObject = ephem.Moon()
        else:
            self.obsObject = ephem.Sun()
        self.pAngle = NP.deg2rad(sunpy.coordinates.sun.P(theDate).to_value())
        self.omegaEarth = coordinates.earth.OMEGA_EARTH.to_value()
        self.update()
        
    def update(self):
        self.obsObject.compute(self.observatory)
        noon = self.obsObject.transit_time
        noonText = str(noon).split(' ')[1].split(':')
        declText = str(self.obsObject.dec).split(':')
        self.culmination = float(noonText[0])*3600. + float(noonText[1])*60. + float(noonText[2]) + float(noon) - int(noon)
        self.declination = NP.deg2rad(float(declText[0]) + NP.sign( int(declText[0]))*(float(declText[1])/60. + float(declText[2])/3600.))
        
    def setDate(self, strDate):
        self.observatory.date = strDate
        self.update()

    def getHourAngle(self, time_sec):
        self.hAngle = self.omegaEarth * (time_sec - self.culmination)
        if self.hAngle > 1.5*NP.pi:
            self.hAngle -= 2*NP.pi
        return self.hAngle
          
    def getDeclination(self):
        return self.declination

    def getPQScale(self, size, FOV, frequency_hz):
        self.cosP = NP.sin(self.hAngle) * NP.cos(self.declination)
        self.cosQ = NP.cos(self.hAngle) * NP.cos(self.declination) * NP.sin(self.observatory.lat) - NP.sin(self.declination) * NP.cos(self.observatory.lat)
        FOV_p = 2.*(constants.c / frequency_hz) / (self.base*NP.sqrt(1. - self.cosP**2.));
        FOV_q = 2.*(constants.c / frequency_hz) / (self.base*NP.sqrt(1. - self.cosQ**2.));
        
        return [int(size*FOV/FOV_p.to_value()), int(size*FOV/FOV_q.to_value())]
        
    def getPQ2HDMatrix(self):
        gP =  NP.arctan(NP.tan(self.hAngle)*NP.sin(self.declination));
        gQ =  NP.arctan(-(NP.sin(self.declination) / NP.tan(self.hAngle) + NP.cos(self.declination) / (NP.sin(self.hAngle)*NP.tan(self.observatory.lat))));
        
        if self.hAngle > 0:
            gQ = NP.pi + gQ;
        g = gP - gQ;
          
        pqMatrix = NP.zeros((3,3))
        pqMatrix[0, 0] =  NP.cos(gP) - NP.cos(g)*NP.cos(gQ)
        pqMatrix[0, 1] = -NP.cos(g)*NP.cos(gP) + NP.cos(gQ)
        pqMatrix[1, 0] =  NP.sin(gP) - NP.cos(g)*NP.sin(gQ)
        pqMatrix[1, 1] = -NP.cos(g)*NP.sin(gP) + NP.sin(gQ)
        pqMatrix /= NP.sin(g)**2.
        pqMatrix[2, 2] = 1.
        return pqMatrix