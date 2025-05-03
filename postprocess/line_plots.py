import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import sys
sys.path.append('..')
from fluid_mechanics.isentropic import isentropic as isen
from scipy.interpolate import interp1d

pwd = os.getcwd()

try:
  files = ['../output/MOC_2d_G1.4000_M1.5000_n0200.txt',
           '../output/MOC_2d_G1.4000_M3.0000_n0200.txt',
           '../output/MOC_2d_G1.4000_M6.0000_n0200.txt']
except:
  os.chdir('..')
  subprocess.call(['./moc_nozzle.py -D 2d -N 200 -I 2 -M 1.5,3,6'])
  files = ['output/MOC_2d_G1.4000_M1.5000_n0200.txt',
           'output/MOC_2d_G1.4000_M3.0000_n0200.txt',
           'output/MOC_2d_G1.4000_M6.0000_n0200.txt']

for f in files:
  # wall data
  wdat = np.loadtxt(f, skiprows=2, max_rows=201)
  xw = wdat[:,0]
  yw = wdat[:,1]
  Mw = wdat[:,2]
  Pw = wdat[:,3]
  Tw = wdat[:,4]
  rhow = wdat[:,5]
  Aw = 2.0*yw

  # centerline data
  cdat = np.loadtxt(f, skiprows=203)
  xc = cdat[:,0]
  yc = cdat[:,1]
  Mc = cdat[:,2]
  Pc = cdat[:,3]
  Tc = cdat[:,4]
  rhoc = cdat[:,5]
  interpA = interp1d(xw,Aw,kind='linear', fill_value='extrapolate')
  Ac = interpA(xc)

  # averaged
  interpM = interp1d(Aw,Mw, kind='linear')
  interpP = interp1d(Aw,Pw, kind='linear')
  interpT = interp1d(Aw,Tw, kind='linear')
  interpD = interp1d(Aw,rhow, kind='linear')
  Ma = np.mean([Mc,interpM(Ac)], axis=0)
  Pa = np.mean([Pc,interpP(Ac)], axis=0)
  Ta = np.mean([Tc,interpT(Ac)], axis=0)
  rhoa = np.mean([rhoc,interpD(Ac)], axis=0)

  # quasi-1d data
  M = isen.Aratio2M(1.4,Aw)
  P = isen.M2Pratio(1.4,M)
  T = isen.M2Tratio(1.4,M)
  rho = isen.M2Rratio(1.4,M)

  # Plot Mach
  fig, ax = plt.subplots()
  ax.plot(xw,M, label='Quasi-1D')
  ax.plot(xw,Mw, label='MOC Wall Profile')
  ax.plot(xc,Mc, label='MOC Centerline')
  ax.plot(xc,Ma, label='MOC Average')
  ax.legend()
  ax.grid()
  ax2 = ax.secondary_xaxis('bottom')
  labels = ax.get_xticklabels()
  label_strings = [label.get_text().replace('−','-') for label in labels]
  labels = [f'\n{interpA(float(n)):.3f}' for n in label_strings]
  ax2.set_xticks(ax.get_xticks())
  ax2.set_xticklabels(labels)
  ax.set_xlabel('\ntop: x, bottom: A/A*')
  ax.set_ylabel('Mach')
  ax.set_title('Comparison of Mach Profiles')
  fig.tight_layout()

  # Plot Pressure Ratio
  fig, ax = plt.subplots()
  ax.plot(xw,P, label='Quasi-1D')
  ax.plot(xw,Pw, label='MOC Wall Profile')
  ax.plot(xc,Pc, label='MOC Centerline')
  ax.plot(xc,Pa, label='MOC Average')
  ax.legend()
  ax.grid()
  ax2 = ax.secondary_xaxis('bottom')
  labels = ax.get_xticklabels()
  label_strings = [label.get_text().replace('−','-') for label in labels]
  labels = [f'\n{interpA(float(n)):.3f}' for n in label_strings]
  ax2.set_xticks(ax.get_xticks())
  ax2.set_xticklabels(labels)
  ax.set_xlabel('\ntop: x, bottom: A/A*')
  ax.set_ylabel(r'$P/P_t$')
  ax.set_title('Comparison of Pressure Ratio Profiles')
  fig.tight_layout()

  # Plot Temperature Ratio
  fig, ax = plt.subplots()
  ax.plot(xw,T, label='Quasi-1D')
  ax.plot(xw,Tw, label='MOC Wall Profile')
  ax.plot(xc,Tc, label='MOC Centerline')
  ax.plot(xc,Ta, label='MOC Average')
  ax.legend()
  ax.grid()
  ax2 = ax.secondary_xaxis('bottom')
  labels = ax.get_xticklabels()
  label_strings = [label.get_text().replace('−','-') for label in labels]
  labels = [f'\n{interpA(float(n)):.3f}' for n in label_strings]
  ax2.set_xticks(ax.get_xticks())
  ax2.set_xticklabels(labels)
  ax.set_xlabel('\ntop: x, bottom: A/A*')
  ax.set_ylabel(r'$T/T_t$')
  ax.set_title('Comparison of Temperature Ratio Profiles')
  fig.tight_layout()

  # Plot Density Ratio
  fig, ax = plt.subplots()
  ax.plot(xw,rho, label='Quasi-1D')
  ax.plot(xw,rhow, label='MOC Wall Profile')
  ax.plot(xc,rhoc, label='MOC Centerline')
  ax.plot(xc,rhoa, label='MOC Average')
  ax.legend()
  ax.grid()
  ax2 = ax.secondary_xaxis('bottom')
  labels = ax.get_xticklabels()
  label_strings = [label.get_text().replace('−','-') for label in labels]
  labels = [f'\n{interpA(float(n)):.3f}' for n in label_strings]
  ax2.set_xticks(ax.get_xticks())
  ax2.set_xticklabels(labels)
  ax.set_xlabel('\ntop: x, bottom: A/A*')
  ax.set_ylabel(r'$\rho/\rho_t$')
  ax.set_title('Comparison of Density Ratio Profiles')
  fig.tight_layout()
  plt.show()
