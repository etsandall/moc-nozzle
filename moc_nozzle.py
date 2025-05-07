#! /usr/bin/env python
#
# Design a supersonic nozzle using Method of Characteristics 
# for CPG, inviscid, steady, isentropic, irrotational flow
# 2D or axisymmetric
#
# Author: Eric Sandall
# Original Creation: 9 July 2022
# Versions: Python 3.7.6
#

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable as mal
import scipy.optimize as opt
from math import asin, pi, tan, sqrt
from numpy import cos, sin
from fluid_mechanics.isentropic import prandtl_meyer as pmr
from fluid_mechanics.isentropic import isentropic as isen

############################
#        MoC Solver        #
############################

deg2rad = pi/180.0
rad2deg = 180.0/pi

class MOC_Nozzle:
    '''
Class for generating 2D or axisymmetric minimum length nozzle contours 
using the method of characteristics. Basic assumptions include:
    1. Steady, isentropic, irrotational flow
    2. Calorically perfect
    3. Initial expansion region is a sharp corner
    4. Sonic line is a straight line at nozzle throat
    5. Throat area A* is unity

Outputs: Data file and optional plot with contour points for upper nozzle contour
inputs:  var      type      default     details
         --------------------------------------
         dim      [str]     'axi'       solver type: '2d' or 'axi'
         gamma    [float]   1.4         specific heat ratio
         M        [float]   2           desired Mach number at nozzle exit
         n        [int]     5           number of characteristics to approximate
         outdir   [str]     'output'    path to output directory
         iplot    [int]     0           flag to save plot contour, 0:no plot,
                                                                   1:show plot only, 
                                                                   2:save plot only,
                                                                   3:save & show plot

General usage (from python)

  from moc_nozzle import MOC_Nozzle as MOC
  moc = MOC(**kwargs)  #kwargs are listed in inputs above

General usage (from linux terminal):

  python moc_nozzle.py [args]
        or
  ./moc_nozzle.py [args]

  e.g. python moc_nozzle.py -h
  e.g. ./moc_nozzle.py -G 1.2 --mach=1.5,2.0,2.5 -n 10 --iplot=2

  [args]
    -d, --default   :   run with default parameters
    -D, --dim       :   2d or axi [str]
    -G, --gamma     :   specific heat ratio [single number or array in Python syntax]
    -h, --help      :   display help
    -I, --iplot     :   flag to save/show contour plots [0:no, 1:show, 2:save, 3:save & show]
    -M, --mach      :   Mach number at nozzle exit [single number or array in Python syntax]
    -N, --n         :   number of characteristics to approximate solution [integer]
    -O, --outdir    :   relative path to save directory for output files/plots [str]
    -R, --r         :   radius of expansion region [single number or array in Python syntax]
                                                   [if r<0, assume minimum length nozzle]
    -t, --test      :   run test cases
    '''

    def __init__(self, dim='axi', gamma=1.4, M=2.0, n=5, outdir='output', iplot=1, r=0.0):
        # Check inputs are valid
        assert dim.upper() in ['2D','AXI'], f'"dim" [str] expects "2d" or "axi", got {dim} ({type(dim)})'
        assert isinstance(gamma, (float, int)), '"gamma" [float] expects a number, got {gamma} ({type(gamma)})'
        assert isinstance(M, (float, int)), '"M" [float] expects a number, got {M} ({type(M)})'
        assert isinstance(n, int), '"n" [int] expects an integer, got {n} ({type(n)})'
        assert isinstance(outdir, str), '"outdir" [str] expects a relative or absolute path string, got {outdir} ({type(outdir)})'
        assert iplot == 0 or iplot == 1 or iplot == 2 or iplot == 3, '"iplot" [int] must be 0, 1, or 2, got {iplot} ({type(iplot)})'
        assert isinstance(r, (float, int)), '"r" [float] expects a number, got {r} ({type(r)})'

        self.dim = dim.upper()
        self.gamma = float(gamma)
        self.Me = float(M)
        self.n = n
        self.outdir = outdir
        self.iplot = iplot
        self.fname_base = f'MOC_{self.dim.lower()}_G{self.gamma:.4f}_M{self.Me:.4f}_n{self.n:04}'
        self.r = float(r)
        if r > 0:
            self.MLN = False
        else:
            self.MLN = True
        
        #x,y-coordinate for start of nozzle expansion
        #solving for top contour only
        #mirror for full 2D or rotate for full axi solution
        self.A0 = 1.0
        self.x0 = 0.0
        if '2D' in self.dim:
            self.y0 = self.A0/2.0
        elif 'AXI' in self.dim:
            self.y0 = sqrt(self.A0/pi)

        # Initialize arrays
        self.x=np.zeros([self.n,self.n])        # x coordinates
        self.y=np.zeros([self.n,self.n])        # y/r coordinates
        self.M=np.zeros([self.n,self.n])        # Mach number
        self.Mu=np.zeros([self.n,self.n])       # Mach angle
        self.theta=np.zeros([self.n,self.n])    # angle relative to horizontal
        self.Nu = np.zeros([self.n,self.n])     # Prandtl-Meyer function
        
        #Constant along right running C- characteristic lines
        self.Km = np.zeros([self.n,self.n])     # K- values: theta + nu
        #Constant along left running C+ characteristic lines
        self.Kp = np.zeros([self.n,self.n])     # K+ values: theta - nu
        
        self.nuMax = pmr.M2nu(self.gamma, self.Me) # Prandtl-Meyer function for design Mach #
        self.thetaMax = self.nuMax/2.0 # Maximum angle at expansion corner of nozzle
        
        # Initialize wall point arrays
        if self.MLN:
          wall_size = self.n + 1
        else:
          wall_size = 2*self.n
        self.xwall = np.zeros([wall_size])
        self.ywall = np.zeros([wall_size])
        self.thetaw = np.zeros([wall_size])
        self.Nuw = np.zeros([wall_size])
        self.Muw = np.zeros([wall_size])
        self.Mw = np.zeros([wall_size])
        self.xwall[0] = self.x0
        self.ywall[0] = self.y0
        self.thetaw[0] = 0.0
        self.Nuw[0] = 0.0
        self.Muw[0] = pmr.nu2mu(self.gamma,self.Nuw[0])
        self.Mw[0] = pmr.nu2M(self.gamma,self.Nuw[0])

        # MOC Solver
        if 'AXI' in self.dim:
            self.MOC_axi()
        elif '2D' in self.dim:
            self.MOC_2D()
        else:
            raise SyntaxError(f'Invalid MOC Solver type.  Should be axi or 2d but got {self.dim.lower()}')

        # Generate output data file
        self.centerline()
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fname = os.path.join(outdir,self.fname_base + '.txt')
        with open(fname, 'w') as f:
          np.savetxt(f, self.wall_data, delimiter='\t', header='Wall Data\nx\ty\tM\tPratio\tTratio\tDratio')
          np.savetxt(f, self.centerline_data, delimiter='\t', header='Centerline Data\nx\ty\tM\tPratio\tTratio\tDratio')

        # Plot
        if self.iplot > 0:
            self.plot_nozzle()

    def MOC_2D(self):
        '''Method of Characteristics solver for 2D nozzle'''

        #Flow data for characteristic lines (1st iteration)
        self.theta[:,0] = np.linspace(self.thetaMax/self.n, self.thetaMax, self.n)
        self.Nu[:,0]    = np.linspace(self.thetaMax/self.n, self.thetaMax, self.n)
        for i in range(self.n):
            self.M[i,0] = pmr.nu2M(self.gamma, self.theta[i,0])
            self.Km[i,0] = self.theta[i,0] + self.Nu[i,0]
            self.Kp[i,0] = self.theta[i,0] - self.Nu[i,0]
            self.Mu[i,0] = asin(1.0/self.M[i,0])*rad2deg
        
        #Flow data for characteristic lines (2: iterations)
        for j in range(1,self.n):
            for i in range(self.n-j):
                if i == 0:
                    self.theta[i,j] = 0.0
                    self.Km[i,j] = self.Km[i+1,j-1]
                    self.Kp[i,j] = 2.0*self.theta[i,j] - self.Km[i,j]
                    self.Nu[i,j] = 0.5*(self.Km[i,j] - self.Kp[i,j])
                    self.Mu[i,j] = pmr.nu2mu(self.gamma,self.Nu[i,j])
                else:
                    self.Km[i,j] = self.Km[i+1,j-1]
                    self.Kp[i,j] = self.Kp[i-1,j]
                    self.theta[i,j] = 0.5*(self.Km[i,j]+self.Kp[i,j])
                    self.Nu[i,j] = 0.5*(self.Km[i,j]-self.Kp[i,j])
                    self.Mu[i,j] = pmr.nu2mu(self.gamma,self.Nu[i,j])
                self.M[i,j] = pmr.nu2M(self.gamma,self.Nu[i,j])
        
        #Characteristic line coordinates (first C+ line)
        self.y[0,0] = 0.0
        self.x[0,0] = self.x0 - self.y0/(tan((self.theta[0,0]-self.Mu[0,0])*deg2rad))
        for i in range(1,self.n):
            mp = tan((self.theta[i-1,0]+self.Mu[i-1,0])*deg2rad)
            mm = tan((self.theta[i,0]-self.Mu[i,0])*deg2rad)
            yi=(self.y[i-1,0]-mp*((self.y0 + self.r - self.r*cos(self.theta[i,0]*deg2rad))/mm-(self.x0 + self.r*sin(self.theta[i,0]*deg2rad))+self.x[i-1,0]))/(1.0-mp/mm)
            xi=(yi-(self.y0 + self.r - self.r*cos(self.theta[i,0]*deg2rad)))/mm+(self.x0 + self.r*sin(self.theta[i,0]*deg2rad))
            self.x[i,0]=xi
            self.y[i,0]=yi
    
        #Characteristic line coordinates (remaining)
        for j in range(1,self.n+1):
            for i in range(0,self.n-j):
                if i==0: #point is on symmetry axis (y=0)
                    yi=0
                    mm=tan((self.theta[i+1,j-1]-self.Mu[i+1,j-1])*deg2rad)
                    xi=(yi-self.y[i+1,j-1]+mm*self.x[i+1,j-1])/mm
                    self.x[i,j]=xi
                    self.y[i,j]=yi
                else:
                    mp=tan((self.theta[i-1,j]+self.Mu[i-1,j])*deg2rad)
                    mm=tan((self.theta[i+1,j-1]-self.Mu[i+1,j-1])*deg2rad)
                    yi=((mm*(-self.y[i-1,j]/mp+self.x[i-1,j]-self.x[i+1,j-1])+
                         self.y[i+1,j-1])/(1-mm/mp))
                    xi=(yi-self.y[i-1,j])/mp+self.x[i-1,j]
                    self.x[i,j]=xi
                    self.y[i,j]=yi

        #Wall data
        self.wall_2D()

    def MOC_axi(self):
        '''Method of Characteristics solver for axisymmetric nozzle'''

        #Initial data at entrance point of throat
        self.thetaL = np.zeros(self.n)
        self.thetaL[:] = np.linspace(self.thetaMax/self.n, self.thetaMax, self.n)
        self.ML = np.ones(self.n)
        self.nuL = np.zeros(self.n)
        self.muL = np.zeros(self.n)
        self.KmL = np.zeros(self.n)
        self.KpL = np.zeros(self.n)
        for i in range(self.n):
            self.nuL[i] = self.thetaL[i]
            self.muL[i] = pmr.nu2mu(self.gamma,self.nuL[i])
            self.KmL[i] = self.thetaL[i] + self.nuL[i]
            self.KpL[i] = self.thetaL[i] - self.nuL[i]
    
        #Flow data for characteristic lines (1st iteration)
        for i in range(self.n):
            if i == 0:
                self.theta[i,0] = 0.0
                self.y[i,0] = 0.0
                self.x[i,0] = ((self.y[i,0]-(self.y0 + self.r - self.r*cos(self.thetaL[i]*deg2rad)))/tan((self.thetaL[i]-
                           self.muL[i])*deg2rad) + (self.x0 + self.r*sin(self.thetaL[i]*deg2rad)))
                self.Km[i,0] = (1.0/(sqrt(self.ML[i]**2.0 - 1.0) -
                          1.0/tan(self.thetaL[i]*deg2rad))/(self.y0 + self.r - self.r*cos(self.thetaL[i]*deg2rad))*(self.y[i,0]-(self.y0 + self.r - self.r*cos(self.thetaL[i]*deg2rad)))+
                          self.KmL[i])
                self.Nu[i,0] = self.Km[i,0] - self.theta[i,0]
                self.Kp[i,0] = self.theta[i,0] - self.Nu[i,0]
                self.Mu[i,0] = pmr.nu2mu(self.gamma,self.Nu[i,0])
                self.M[i,0] = pmr.nu2M(self.gamma,self.Nu[i,0])
            else:
                self.x[i,0] = ((tan((self.thetaL[i]-self.muL[i])*deg2rad)*(self.x0 + self.r*sin(self.thetaL[i]*deg2rad)) -
                     tan((self.theta[i-1,0]+self.Mu[i-1,0])*deg2rad)*self.x[i-1,0]+
                     self.y[i-1,0] - (self.y0 + self.r - self.r*cos(self.thetaL[i]*deg2rad)))/(tan((self.thetaL[i]-self.muL[i])*deg2rad)-
                     tan((self.theta[i-1,0]+self.Mu[i-1,0])*deg2rad)))
                self.y[i,0] = (tan((self.thetaL[i]-
                     self.muL[i])*deg2rad)*(self.x[i,0]-(self.x0 + self.r*sin(self.thetaL[i]*deg2rad))) + (self.y0 + self.r - self.r*cos(self.thetaL[i]*deg2rad)))
                if i == 1:
                    self.Nu[i,0] = ((1.0/(sqrt(self.ML[i]**2.0-1.0)-
                        1.0/tan(self.thetaL[i]*deg2rad))/(self.y0 + self.r - self.r*cos(self.thetaL[i]*deg2rad))*(self.y[i,0]-
                        (self.y0 + self.r - self.r*cos(self.thetaL[i]*deg2rad)))+(self.thetaL[i]+self.nuL[i])-(self.theta[i-1,0]-
                        self.Nu[i-1,0]))/2.0)
                    self.theta[i,0] = (self.theta[i-1,0]-self.Nu[i-1,0])+self.Nu[i,0]
                else:
                    self.Nu[i,0] = ((1.0/(sqrt(self.ML[i]**2.0-1.0)-
                        1.0/tan(self.thetaL[i]*deg2rad))/(self.y0 + self.r - self.r*cos(self.thetaL[i]*deg2rad))*(self.y[i,0]-
                        (self.y0 + self.r - self.r*cos(self.thetaL[i]*deg2rad)))+(self.thetaL[i]+self.nuL[i])+1.0/(sqrt(self.M[i-1,0]**2.0-
                        1.0)+1.0/tan(self.theta[i-1,0]*deg2rad))/self.y[i-1,0]*
                        (self.y[i,0]-self.y[i-1,0])-(self.theta[i-1,0]-
                        self.Nu[i-1,0]))/2.0)
                    self.theta[i,0] = (-1.0/(sqrt(self.M[i-1,0]**2.0-1.0)+
                        1.0/tan(self.theta[i-1,0]*deg2rad))/self.y[i-1,0]*
                        (self.y[i,0]-self.y[i-1,0])+(self.theta[i-1,0]-
                        self.Nu[i-1,0])+self.Nu[i,0])
                self.Km[i,0] = self.theta[i,0] + self.Nu[i,0]
                self.Kp[i,0] = self.theta[i,0] - self.Nu[i,0]
                self.Mu[i,0] = pmr.nu2mu(self.gamma,self.Nu[i,0])
                self.M[i,0] = pmr.nu2M(self.gamma,self.Nu[i,0])
    
        #Flow data for characteristic lines (2: iterations)
        for j in range(1,self.n):
            for i in range(self.n-j):
                if i == 0: #Center line (symmetry line)
                    self.theta[i,j] = 0.0
                    self.y[i,j] = 0.0
                    self.x[i,j] = ((self.y[i,j]-self.y[i+1,j-1])/tan((self.theta[i+1,
                        j-1]-self.Mu[i+1,j-1])*deg2rad) + self.x[i+1,j-1])
                    self.Km[i,j] = (1.0/(sqrt(self.M[i+1,j-1]**2.0 - 1.0) -
                        1.0/tan(self.theta[i+1,j-1]*deg2rad))/self.y[i+1, j-1]*
                        (self.y[i,j]-self.y[i+1,j-1]) + self.Km[i+1,j-1])
                    self.Nu[i,j] = self.Km[i,j] - self.theta[i,j]
                    self.Kp[i,j] = self.theta[i,j] - self.Nu[i,j]
                    self.Mu[i,j] = pmr.nu2mu(self.gamma,self.Nu[i,j])
                    self.M[i,j] = pmr.nu2M(self.gamma,self.Nu[i,j])
                else:
                    self.x[i,j] = ((tan((self.theta[i+1,j-1]-self.Mu[i+1,
                        j-1])*deg2rad)*self.x[i+1,j-1]-tan((self.theta[i-1,
                        j]+self.Mu[i-1,j])*deg2rad)*self.x[i-1,j]+
                        self.y[i-1,j]-self.y[i+1,j-1])/(tan((self.theta[i+1,j-1]-
                        self.Mu[i+1,j-1])*deg2rad)-tan((self.theta[i-1,j]+
                        self.Mu[i-1,j])*deg2rad)))
                    self.y[i,j] = (tan((self.theta[i+1,j-1]-self.Mu[i+1,
                        j-1])*deg2rad)*(self.x[i,j]-self.x[i+1,j-1])+
                        self.y[i+1,j-1])
                    if i == 1:
                        self.Nu[i,j] = ((1.0/(sqrt(self.M[i+1,j-1]**2.0-1.0)-1.0/
                            tan(self.theta[i+1,j-1]*deg2rad))/self.y[i+1,j-1]*
                            (self.y[i,j]-self.y[i+1,j-1])+(self.theta[i+1,j-1]+
                            self.Nu[i+1,j-1])-(self.theta[i-1,j]-self.Nu[i-1,j]))/2.0)
                        self.theta[i,j] = (self.theta[i-1,j]-self.Nu[i-1,j])+self.Nu[i,j]
                    else:
                        self.Nu[i,j] = ((1.0/(sqrt(self.M[i+1,j-1]**2.0-1.0)-1.0/
                            tan(self.theta[i+1,j-1]*deg2rad))/self.y[i+1,j-1]*
                            (self.y[i,j]-self.y[i+1,j-1])+(self.theta[i+1,j-1]+
                            self.Nu[i+1,j-1])+1.0/(sqrt(self.M[i-1,j]**2.0-1.0)+
                            1.0/tan(self.theta[i-1,j]*deg2rad))/self.y[i-1,j]*
                            (self.y[i,j]-self.y[i-1,j])-(self.theta[i-1,j]-
                            self.Nu[i-1,j]))/2.0)
                        self.theta[i,j] = (-1.0/(sqrt(self.M[i-1,j]**2.0-1.0)+
                            1.0/tan(self.theta[i-1,j]*deg2rad))/self.y[i-1,
                            j]*(self.y[i,j]-self.y[i-1,j])+(self.theta[i-1,j]-
                            self.Nu[i-1,j])+self.Nu[i,j])
                    self.Km[i,j] = self.theta[i,j] + self.Nu[i,j]
                    self.Kp[i,j] = self.theta[i,j] - self.Nu[i,j]
                    self.Mu[i,j] = pmr.nu2mu(self.gamma,self.Nu[i,j])
                    self.M[i,j] = pmr.nu2M(self.gamma,self.Nu[i,j])
    
        #Wall data
        self.wall_axi()
    
    def wall_2D(self):
        '''Calculate wall points for 2D nozzle'''

        if self.MLN:
            start = 0
            self.thetaw[0] = self.thetaMax
            self.Nuw[0] = self.thetaMax
        else:
            start = self.n - 1
            self.thetaw[1:self.n+1] = np.linspace(self.thetaMax/self.n, self.thetaMax, self.n)
            self.Nuw[1:self.n+1] = np.linspace(self.thetaMax/self.n, self.thetaMax, self.n)
            self.xwall[1:self.n+1] = self.x0 + self.r*sin(self.thetaw[1:self.n+1]*deg2rad)
            self.ywall[1:self.n+1] = self.y0 + self.r - self.r*cos(self.thetaw[1:self.n+1]*deg2rad)

        #Wall angles
        for j in range(self.n):
            self.thetaw[start+j+1] = self.theta[self.n-j-1,j]
            self.Nuw[start+j+1] = self.Nu[self.n-j-1,j]
    
        #Wall points
        for j in range(1,self.n+1):
    	      #average wall slope between two points
            mw=((tan((self.thetaw[start+j-1])*deg2rad)+
    	         tan((self.thetaw[start+j])*deg2rad))/2.0); 
            mp=(tan((self.theta[self.n-j,j-1]+
                self.Mu[self.n-j,j-1])*deg2rad))
            yj=((mp*(-self.ywall[start+j-1]/mw+self.xwall[start+j-1]-self.x[self.n-j,j-1])+
                 self.y[self.n-j,j-1])/(1-mp/mw))
            xj=(yj-self.ywall[start+j-1])/mw+self.xwall[start+j-1]
            self.xwall[start+j]=xj
            self.ywall[start+j]=yj
        assert all(np.diff(self.xwall) >= 0.0), 'Intersecting characteristics in simple region in expanding section.'

        #Organize data
        self.Mw = pmr.nu2M(self.gamma, self.Nuw)
        Pratiow = isen.M2Pratio(self.gamma, self.Mw)
        Tratiow = isen.M2Tratio(self.gamma, self.Mw)
        Rratiow = isen.M2Rratio(self.gamma, self.Mw)
        wall_data = [self.xwall, self.ywall, self.Mw, Pratiow, Tratiow, Rratiow]
        self.wall_data = np.transpose(wall_data)

    def wall_axi(self):
        '''Calculate wall points for axisymmetric nozzle'''

        if self.MLN:
            start = 0
            self.thetaw[0] = self.thetaMax
            self.Nuw[0] = self.thetaMax
        else:
            start = self.n - 1
            self.thetaw[1:self.n+1] = np.linspace(self.thetaMax/self.n, self.thetaMax, self.n)
            self.Nuw[1:self.n+1] = np.linspace(self.thetaMax/self.n, self.thetaMax, self.n)
            self.xwall[1:self.n+1] = self.x0 + self.r*sin(self.thetaw[1:self.n+1]*deg2rad)
            self.ywall[1:self.n+1] = self.y0 + self.r - self.r*cos(self.thetaw[1:self.n+1]*deg2rad)

        for j in range(1,self.n+1):
            def myFunctions(z):
                [X,self.Km] = z
                eq = np.empty((2))
                eq[0] = (tan((self.thetaw[start+j-1]+
                    self.thetaw[start+j])/4.0*deg2rad)*(X-self.xwall[start+j-1])-
                    (self.Km-self.ywall[start+j-1]))
                eq[1] = (tan((self.theta[self.n-j,j-1]+
                    self.Mu[self.n-j,j-1])*deg2rad)*(X-
                    self.x[self.n-j,j-1])-(self.Km-self.y[self.n-j-1,j-1]))
                return eq
    
            self.thetaw[start+j] = self.theta[self.n-j,j-1]
            self.Mw[start+j] = self.M[self.n-j,j-1]
            self.Nuw[start+j] = self.Nu[self.n-j,j-1]
            self.Muw[start+j] = self.Mu[self.n-j,j-1]
    
            guess = np.array([self.xwall[start+j-1],self.ywall[start+j-1]])
            sol = opt.root(myFunctions, guess, method='hybr')
            self.xwall[start+j] = sol.x[0]
            self.ywall[start+j] = sol.x[1]
        assert all(np.diff(self.xwall) >= 0.0), 'Intersecting characteristics in simple region in expanding section.'

        #Organize data
        Pratiow = isen.M2Pratio(self.gamma, self.Mw)
        Tratiow = isen.M2Tratio(self.gamma, self.Mw)
        Rratiow = isen.M2Rratio(self.gamma, self.Mw)
        wall_data = [self.xwall, self.ywall, self.Mw, Pratiow, Tratiow, Rratiow]
        self.wall_data = np.transpose(wall_data)

    def centerline(self):
        centerline = np.where(abs(self.y) < 1e-6)
        xcenter = np.concatenate([[0.0], self.x[0],[self.xwall[-1]]])
        ycenter = np.zeros(len(xcenter))
        Mcenter = np.concatenate([[1.0], self.M[0], [self.M[0][-1]]])
        Pratiocenter = isen.M2Pratio(self.gamma, Mcenter)
        Tratiocenter = isen.M2Tratio(self.gamma, Mcenter)
        Rratiocenter = isen.M2Rratio(self.gamma, Mcenter)
        centerline_data = [xcenter, ycenter, Mcenter, Pratiocenter, Tratiocenter, Rratiocenter]
        self.centerline_data = np.transpose(centerline_data)

    def plot_nozzle(self):
        '''Plot nozzle contour and characteristic lines'''
        fig, ax = plt.subplots()

        #plot wall geometry
        ax.plot(self.xwall,self.ywall,'k')
        
        #plot first characteristics from nozzle throat
        for i in range(self.n):
            ax.plot([self.x0 + self.r*sin(self.theta[i,0]*deg2rad),self.x[i,0]],[self.y0 + self.r - self.r*cos(self.theta[i,0]*deg2rad), self.y[i,0]],'b',linewidth=0.5)
        
        #plot characteristics from wall
        if self.MLN:
            start = 0
        else:
            start = self.n - 1
        for j in range(1,self.n+1):
            ax.plot([self.x[self.n-j,j-1],self.xwall[start+j]],[self.y[self.n-j,j-1],
                     self.ywall[start+j]],'b',linewidth=0.5)
        
        #plot inner characteristics
        for i in range(self.n-1):
            for j in range(self.n-i-1):
                ax.plot([self.x[i,j],self.x[i+1,j]],[self.y[i,j],
                         self.y[i+1,j]],'b',linewidth=0.5)
                ax.plot([self.x[i+1,j],self.x[i,j+1]],[self.y[i+1,j],
                         self.y[i,j+1]],'b',linewidth=0.5)

        #Plot settings
        outdir = self.outdir
        if '2D' in self.dim:
            ylabel = 'Height [y]'
            pltTitle = 'Minimum Length Nozzle (2D)'
            Aratio = self.ywall[-1]*2.0/(self.ywall[0]*2.0)
        elif 'AXI' in self.dim:
            ylabel = 'Radius [r]'
            pltTitle = 'Minimum Length Nozzle (Axisymmetric)'
            Aratio = pi*self.ywall[-1]**2.0/(pi*self.ywall[0]**2.0)
        pltTitle += f'\nMach={self.Me} | γ={self.gamma} | {self.n} characteristics | A/A*={Aratio:.3f}'
        plotname = os.path.join(outdir,'figs',self.fname_base + '.png')
        ax.set_xlabel('Length [x]')
        ax.set_ylabel(ylabel)
        ax.set_title(pltTitle)
        ax.set_aspect(1)
        ax.set_xlim(xmin=self.xwall[0], xmax=1.05*self.xwall[-1])
        ax.set_ylim(ymin=0, ymax=1.05*np.max(self.ywall))
        ax.grid(True)
        if self.iplot >= 2:
            if not os.path.exists(os.path.dirname(plotname)):
                os.makedirs(os.path.dirname(plotname))
            fig.savefig(plotname)

        # Contour plots of thermodynamic variables
        # This is pretty ugly - just trying to hack the data together in a way
        # to show the contours nicely.  But it works! (hopefully)

        # Combine wall data with non-simple region data
        xfull = np.vstack((self.x, np.zeros([1,len(self.x)])))
        yfull = np.vstack((self.y, np.zeros([1,len(self.y)])))
        Mfull = np.vstack((self.M, np.zeros([1,len(self.M)])))
        for i in range(self.n):
            xfull[self.n-i,i] = self.xwall[start+i+1]
            yfull[self.n-i,i] = self.ywall[start+i+1]
            Mfull[self.n-i,i] = self.Mw[start+i+1]

        # Add data from throat
        prex = np.zeros([self.n+1,1])
        prey = self.ywall[0]*np.ones([self.n+1,1])
        if not self.MLN:
            prex[1:,0] = self.x0 + self.r*sin(self.thetaw[:self.n]*deg2rad)
            prey[1:,0] = self.y0 + self.r - self.r*cos(self.thetaw[:self.n]*deg2rad)
        xfull = np.concatenate([prex,xfull], axis=1)
        yfull = np.concatenate([prey,yfull], axis=1)
        yfull[0,0] = 0.0
        Mfull = np.concatenate([np.ones([self.n+1,1]),Mfull], axis=1)
        Mfull[1:,0] = Mfull[:-1,1]

        # Add data from exit
        xfull = np.concatenate([xfull, np.zeros([self.n+1,1])], axis=1)
        yfull = np.concatenate([yfull, np.zeros([self.n+1,1])], axis=1)
        Mfull = np.concatenate([Mfull, np.full([self.n+1,1],np.nan)], axis=1)
        xfull[0,-1] = self.xwall[-1]
        yfull[0,-1] = 0.0
        Mfull[0,-1] = self.Mw[-1]
        mask = np.fliplr(np.tri(Mfull.shape[0], Mfull.shape[1],k=-1))
        Mfull = np.ma.masked_array(Mfull, mask=mask)
        Pratio = np.ma.masked_array(isen.M2Pratio(self.gamma, Mfull), mask=mask)

        fig, ax = plt.subplots()
        cf = ax.contourf(xfull,yfull,Mfull, cmap='coolwarm', levels=200)
        pltTitleM = pltTitle + '\nMach'
        ax.set_aspect(1)
        ax.set_title(pltTitleM)
        ax.set_xlabel('Length [x]')
        ax.set_ylabel(ylabel)
        divider = mal(ax)
        cax = divider.append_axes('right', size='4%', pad=0.05)
        cbar = fig.colorbar(cf, cax=cax)
        cbar.set_ticks(np.linspace(Mfull.min(), Mfull.max(), 5))
        fig.tight_layout()
        if self.iplot >= 2:
            fig.savefig(plotname.replace('.png','_mach.png'))

        fig, ax = plt.subplots()
        cf = ax.contourf(xfull,yfull,Pratio, cmap='coolwarm', levels=200)
        pltTitleP = pltTitle + '\n' + r'$P/P_t$'
        ax.set_aspect(1)
        ax.set_title(pltTitleP)
        ax.set_xlabel('Length [x]')
        ax.set_ylabel(ylabel)
        divider = mal(ax)
        cax = divider.append_axes('right', size='4%', pad=0.05)
        cbar = fig.colorbar(cf, cax=cax)
        cbar.set_ticks(np.linspace(Pratio.min(), Pratio.max(), 5))
        fig.tight_layout()
        if self.iplot >= 2:
            fig.savefig(plotname.replace('.png','_pressure.png'))

        if self.iplot == 1 or self.iplot == 3:
            plt.show()
        plt.close(fig)

    def _test(self):
        '''Run MOC_Nozzle test cases'''

        outdir = 'tests'
        print('Running MOC_Nozzle tests...')
        print('2D Results:')
        print('\tMach\tn_chars\tA/A*\tMOC A/A*\terror')
        for M in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
            if M == 1.5: Aratio = 1.176
            elif M == 2.0: Aratio = 1.687
            elif M == 2.5: Aratio = 2.637
            elif M == 3.0: Aratio = 4.235
            elif M == 4.0: Aratio = 10.72
            elif M == 5.0: Aratio = 25.00
            for n in [5, 10, 20, 50]:
                nozzle = MOC_Nozzle('2d', 1.4, M, n, outdir, iplot=0)
                AR_sim = nozzle.ywall[-1]*2.0/(nozzle.ywall[0]*2.0)
                print(f"\t{M:.1f}\t{n}\t{Aratio:.3f}\t{AR_sim:.6f}\t{abs(AR_sim-Aratio)/Aratio*100.0:.3f}%")
        print('Axisymmetric Results:')
        print('\tMach\tn_chars\tA/A*\tMOC A/A*\terror')
        for M in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
            if M == 1.5: Aratio = 1.176
            elif M == 2.0: Aratio = 1.687
            elif M == 2.5: Aratio = 2.637
            elif M == 3.0: Aratio = 4.235
            elif M == 4.0: Aratio = 10.72
            elif M == 5.0: Aratio = 25.00
            for n in [5, 10, 20, 50]:
                nozzle = MOC_Nozzle('axi', 1.4, M, n, outdir, iplot=0)
                AR_sim = pi*nozzle.ywall[-1]**2.0/(pi*nozzle.ywall[0]**2.0)
                print(f"\t{M:.1f}\t{n}\t{Aratio:.3f}\t{AR_sim:.6f}\t{abs(AR_sim-Aratio)/Aratio*100.0:.3f}%")
        test_plot = MOC_Nozzle('2d', 1.4, 2.2, 20, outdir, 3)
        test_plot = MOC_Nozzle('axi', 1.4, 2.2, 20, outdir, 3)
        print('Tests complete.')

    def Sauer(self):
        '''Sauer solution for sonic line at nozzle throat (not implemented)'''

        r = np.linspace(0,self.y0,self.n)
        in_curve = 2.0*self.y0
        alpha = sqrt(2.0/((self.gamma+1.0)*in_curve*self.y0))
        eta = self.y0/8.0*sqrt(2.0*(self.gamma+1.0)*self.y0/in_curve)
        x_star = -eta
        u = alpha*x_star+(self.gamma+1)/4.0*alpha**2.0*self.y0**2.0
        v = ((self.gamma+1.0)/2.0*(alpha**2.0*x_star*self.y0)+
              alpha**3.0*self.y0**3.0*(self.gamma+1.0)**2.0/16.0)
        mach = sqrt((1+u)**2.0+v**2.0)
        x = []
        r = np.linspace(0,self.y0,self.n)
        for R in r:
            x.append(-(self.gamma+1)/4.0*alpha*R**2.0)
        x = x - np.min(x)
        plt.scatter(x,r)
        plt.axis('equal')
        plt.show()
        plt.close()

if __name__ == '__main__':
    print('\nMOC Nozzle by Eric Sandall')
    print('--------------------------')

    # Parsing bash inputs
    if len(sys.argv) <= 1:
        help(MOC_Nozzle)
    elif any(arg in sys.argv[1:] for arg in ['-h','--help']):
        help(MOC_Nozzle)
    elif set(['-d','--default']).intersection(sys.argv[1:]):
        print('Running with default parameters.')
        print(f"dim='axi', gamma=1.4, M=2.0, n=5, outdir='output', iplot=1, r=-1.0")
        moc = MOC_Nozzle()
    elif set(['-t','--test']).intersection(sys.argv[1:]):
        print('Running test cases.')
        MOC_Nozzle._test(MOC_Nozzle)
    else:
        # set everything initially to default values, then update if specified
        D = ['axi']
        G = [1.4]
        I = 1
        M = [2.0]
        N = [5]
        O = 'output'
        R = [-1.0]

        arg = sys.argv[1:]
        i = 0
        while i < len(arg):
            if '-D' == arg[i]:
                D = arg[i+1].split(',')
                i += 1
            elif '--dim' == arg[i].split('=')[0]:
                D = arg[i].split('=')[-1].split(',')
            elif '-G' == arg[i]:
                G = [float(a) for a in arg[i+1].split(',')]
                i += 1
            elif '--gamma' == arg[i].split('=')[0]:
                G = [float(a) for a in arg[i].split('=')[-1].split(',')]
            elif '-I' == arg[i]:
                I = int(arg[i+1])
                i += 1
            elif '--iplot' == arg[i].split('=')[0]:
                I = int(arg[i].split('=')[-1])
            elif '-M' == arg[i]:
                M = [float(a) for a in arg[i+1].split(',')]
                i += 1
            elif '--mach' == arg[i].split('=')[0]:
                M = [float(a) for a in arg[i].split('=')[-1].split(',')]
            elif '-N' == arg[i]:
                N = [int(a) for a in arg[i+1].split(',')]
                i += 1
            elif '--n' == arg[i].split('=')[0]:
                N = [int(a) for a in arg[i].split('=')[-1].split(',')]
            elif '-O' == arg[i]:
                O = str(arg[i+1])
                i += 1
            elif '--outdir' == arg[i].split('=')[0]:
                O = str(arg[i].split('=')[-1])
            elif '-R' == arg[i]:
                R = [float(a) for a in arg[i+1].split(',')]
                i += 1
            elif '--r' == arg[i].split('=')[0]:
                R = [float(a) for a in arg[i].split('=')[-1].split(',')]
            else:
                raise ValueError(f'Unknown input \"{arg[i]}\". Use -h flag for usage help.')
            i += 1
        print('Running MOC_Nozzle with the following parameters:')
        print(f'dim={D}, gamma={G}, M={M}, n={N}, outdir={O}, iplot={I}, r={R}')
        print('\nIf these are not the desired values, check your syntax. Try -h or --help')
        for d in D:
            for g in G:
                for m in M:
                    for n in N:
                      for r in R:
                        MOC_Nozzle(d,g,m,n,O,I,r)
