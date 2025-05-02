import numpy as np
from scipy.optimize import fsolve

def M2Aratio(k, M):
  '''Given specific heat ratio and Mach number, compute A/A*'''
  return ((k+1.0)/2.0)**(-(k+1.0)/(2.0*(k-1.0)))*(1.0 + (k-1.0)/2.0*M**2.0)**((k+1.0)/(2.0*(k-1.0)))/M

def Aratio2M(k, Aratio, supersonic=True):
  '''Given specific heat ratio and A/A*, compute Mach number (flag determines subsonic or supersonic solution'''
  def f(M):
    return Aratio - M2Aratio(k, M)
  if supersonic:
    M = fsolve(f, 5.0, xtol=1e-12)[0]
    assert M >= 1.0, "Mach number is subsonic."
  else:
    M = fsolve(f, 0.1, xtol=1e-12)[0]
    assert M <= 1.0, "Mach number is supersonic."
  return M

def M2Pratio(k, M):
  '''Given specific heat ratio and Mach number, compute P/P_t'''
  return (1.0 + (k-1.0)/2.0*M**2.0)**(-k/(k-1.0))

def Pratio2M(k, Pratio):
  '''Given specific heat ratio and P/P_t, compute Mach number'''
  return np.sqrt((Pratio**((k-1.0)/(-k)) - 1.0)*2.0/(k-1.0))

def M2Tratio(k, M):
  '''Given specific heat ratio and Mach number, compute T/T_t'''
  return (1.0 + (k-1.0)/2.0*M**2.0)**-1.0

def Tratio2M(k, Tratio):
  '''Given specific heat ratio and P/P_t, compute Mach number'''
  return np.sqrt((Tratio**-1.0 - 1.0)*2.0/(k-1.0))

def M2Rratio(k, M):
  '''Given specific heat ratio and Mach number, compute rho/rho_t'''
  return (1.0 + (k-1.0)/2.0*M**2.0)**(-1.0/(k-1.0))

def Rratio2M(k, Rratio):
  '''Given specific heat ratio and rho/rho_t, compute Mach number'''
  return np.sqrt((Rratio**((k-1.0)/(-1.0)) - 1.0)*2.0/(k-1.0))

if __name__ == '__main__':
  print('Running tests...', end=" ")
  gamma = 1.4
  Msup = 2.324
  Msub = 0.743
  tol = 1e-9
  Msup2 = Aratio2M(gamma, M2Aratio(gamma, Msup))
  Msub2 = Aratio2M(gamma, M2Aratio(gamma, Msub), False)
  assert abs(Msup - Msup2)/Msup <= tol, "\nSupersonic Mach from area ratio doesn't match"
  assert abs(Msub - Msub2)/Msup <= tol, "\nSubsonic Mach from area ratio doesn't match"

  Aratio_sup = 2.241467302147679
  Aratio_sub = 1.066393050442894
  Aratio_sup2 = M2Aratio(gamma, Msup)
  Aratio_sub2 = M2Aratio(gamma, Msub)
  assert abs(Aratio_sup - Aratio_sup2)/Aratio_sup <= tol, "\nSupersonic area ratio doesn't match"
  assert abs(Aratio_sub - Aratio_sub2)/Aratio_sup <= tol, "\nSubsonic area ratio doesn't match"

  Pratio = 0.07702570558120181
  Pratio2 = M2Pratio(gamma, Msup)
  Msup2 = Pratio2M(gamma, Pratio2)
  assert abs(Msup - Msup2)/Msup <= tol, "\nMach from pressure ratio doesn't match"
  assert abs(Pratio - Pratio2)/Pratio <= tol, "\nPressure ratio doesn't match"
  
  Tratio = 0.4807241166598213
  Tratio2 = M2Tratio(gamma, Msup)
  Msup2 = Tratio2M(gamma, Tratio2)
  assert abs(Msup - Msup2)/Msup <= tol, "\nMach from temperature ratio doesn't match"
  assert abs(Tratio - Tratio2)/Tratio <= tol, "\nTemperature ratio doesn't match"
  
  Rratio = 0.16022850302662917
  Rratio2 = M2Rratio(gamma, Msup)
  Msup2 = Rratio2M(gamma, Rratio2)
  assert abs(Msup - Msup2)/Msup <= tol, "\nMach from density ratio doesn't match"
  assert abs(Rratio - Rratio2)/Rratio <= tol, "\nDensity ratio doesn't match"
  print('Passed!')
