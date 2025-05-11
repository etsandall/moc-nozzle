import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import sys
sys.path.append('..')
from fluid_mechanics.isentropic import isentropic as isen
from scipy.interpolate import interp1d

pwd = os.getcwd()
files = ['../output/MOC_2d_G1.4000_M1.5000_n0200.txt',
         '../output/MOC_2d_G1.4000_M3.0000_n0200.txt',
         '../output/MOC_2d_G1.4000_M6.0000_n0200.txt']

try:
  np.loadtxt(files[0])
except:
  os.chdir('..')
  subprocess.call('./moc_nozzle.py -D 2d -N 200 -I 2 -M 1.5,3,6 -R 0'.split(' '))
  os.chdir(pwd)

for f in files:
  # wall data
  wdat = np.loadtxt(f, skiprows=2, max_rows=201)
  xw = wdat[:,0]
  yw = wdat[:,1]
  Mw = wdat[:,2]
  Pw = wdat[:,3]
  Tw = wdat[:,4]
  Dw = wdat[:,5]
  Aw = 2.0*yw
  interpAw = interp1d(xw,Aw,kind='linear',fill_value='extrapolate')
  interpMw = interp1d(xw,Mw,kind='linear',fill_value='extrapolate')
  interpPw = interp1d(xw,Pw,kind='linear',fill_value='extrapolate')
  interpTw = interp1d(xw,Tw,kind='linear',fill_value='extrapolate')
  interpDw = interp1d(xw,Dw,kind='linear',fill_value='extrapolate')

  # centerline data
  cdat = np.loadtxt(f, skiprows=203)
  xc = cdat[:,0]
  yc = cdat[:,1]
  Mc = cdat[:,2]
  Pc = cdat[:,3]
  Tc = cdat[:,4]
  Dc = cdat[:,5]
  Ac = interpAw(xc)
  interpAc = interp1d(xc,Ac,kind='linear',fill_value='extrapolate')
  interpMc = interp1d(xc,Mc,kind='linear',fill_value='extrapolate')
  interpPc = interp1d(xc,Pc,kind='linear',fill_value='extrapolate')
  interpTc = interp1d(xc,Tc,kind='linear',fill_value='extrapolate')
  interpDc = interp1d(xc,Dc,kind='linear',fill_value='extrapolate')

  # combine A arrays and interpolate all values
  x = np.sort(np.concatenate([xw,xc]))
  Aw = interpAw(x)
  Mw = interpMw(x)
  Pw = interpPw(x)
  Tw = interpTw(x)
  Dw = interpDw(x)
  Ac = interpAc(x)
  Mc = interpMc(x)
  Pc = interpPc(x)
  Tc = interpTc(x)
  Dc = interpDc(x)
  
  # averaged
  Ma = np.mean([Mc,Mw], axis=0)
  Pa = np.mean([Pc,Pw], axis=0)
  Ta = np.mean([Tc,Tw], axis=0)
  Da = np.mean([Dc,Dw], axis=0)

  # quasi-1d data
  Aw_filter = Aw[~np.isnan(Aw)]
  x_filter = x[~np.isnan(Aw)]
  M = isen.Aratio2M(1.4,Aw_filter)
  P = isen.M2Pratio(1.4,M)
  T = isen.M2Tratio(1.4,M)
  D = isen.M2Rratio(1.4,M)

  # Plot Mach
  fig, ax = plt.subplots()
  ax.plot(x_filter,M, label='Quasi-1D')
  ax.plot(x,Mw, label='MOC Wall Profile')
  ax.plot(x,Mc, label='MOC Centerline')
  ax.plot(x,Ma, label='MOC Average')
  ax.legend()
  ax.grid()
  ax2 = ax.secondary_xaxis('bottom')
  labels = ax.get_xticklabels()
  label_strings = [label.get_text().replace('−','-') for label in labels]
  labels = [f'\n{interpAw(float(n)):.3f}' for n in label_strings]
  ax2.set_xticks(ax.get_xticks())
  ax2.set_xticklabels(labels)
  ax.set_xlabel('\ntop: x, bottom: A/A*')
  ax.set_ylabel('Mach')
  ax.set_title('Comparison of Mach Profiles')
  fig.tight_layout()

  # Plot Pressure Ratio
  fig, ax = plt.subplots()
  ax.plot(x_filter,P, label='Quasi-1D')
  ax.plot(x,Pw, label='MOC Wall Profile')
  ax.plot(x,Pc, label='MOC Centerline')
  ax.plot(x,Pa, label='MOC Average')
  ax.legend()
  ax.grid()
  ax2 = ax.secondary_xaxis('bottom')
  labels = ax.get_xticklabels()
  label_strings = [label.get_text().replace('−','-') for label in labels]
  labels = [f'\n{interpAw(float(n)):.3f}' for n in label_strings]
  ax2.set_xticks(ax.get_xticks())
  ax2.set_xticklabels(labels)
  ax.set_xlabel('\ntop: x, bottom: A/A*')
  ax.set_ylabel(r'$P/P_t$')
  ax.set_title('Comparison of Pressure Ratio Profiles')
  fig.tight_layout()

  # Plot Temperature Ratio
  fig, ax = plt.subplots()
  ax.plot(x_filter,T, label='Quasi-1D')
  ax.plot(x,Tw, label='MOC Wall Profile')
  ax.plot(x,Tc, label='MOC Centerline')
  ax.plot(x,Ta, label='MOC Average')
  ax.legend()
  ax.grid()
  ax2 = ax.secondary_xaxis('bottom')
  labels = ax.get_xticklabels()
  label_strings = [label.get_text().replace('−','-') for label in labels]
  labels = [f'\n{interpAw(float(n)):.3f}' for n in label_strings]
  ax2.set_xticks(ax.get_xticks())
  ax2.set_xticklabels(labels)
  ax.set_xlabel('\ntop: x, bottom: A/A*')
  ax.set_ylabel(r'$T/T_t$')
  ax.set_title('Comparison of Temperature Ratio Profiles')
  fig.tight_layout()

  # Plot Density Ratio
  fig, ax = plt.subplots()
  ax.plot(x_filter,D, label='Quasi-1D')
  ax.plot(x,Dw, label='MOC Wall Profile')
  ax.plot(x,Dc, label='MOC Centerline')
  ax.plot(x,Da, label='MOC Average')
  ax.legend()
  ax.grid()
  ax2 = ax.secondary_xaxis('bottom')
  labels = ax.get_xticklabels()
  label_strings = [label.get_text().replace('−','-') for label in labels]
  labels = [f'\n{interpAw(float(n)):.3f}' for n in label_strings]
  ax2.set_xticks(ax.get_xticks())
  ax2.set_xticklabels(labels)
  ax.set_xlabel('\ntop: x, bottom: A/A*')
  ax.set_ylabel(r'$\rho/\rho_t$')
  ax.set_title('Comparison of Density Ratio Profiles')
  fig.tight_layout()
  plt.show()
