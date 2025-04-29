
# Author: Eric Sandall
# Original Creation: 9 July 2022
# Versions: Python 3.7.6

from math import pi, sqrt, atan, asin

############################
# Prandtl-Meyer Relations  #
############################

def M2nu(G, M):
    '''Prandtl-Meyer relation: Mach number -> PM function'''

    # Outputs: nu [Prandtl-Meyer function] in degrees
    # inputs:  G  [specific heat ratio]
    #          M  [Mach number]
    return (((sqrt((G+1.0)/(G-1.0)))*atan(sqrt((G-1.0)*(M**2.0 - 1.0)/(G+1.0))) - 
            atan(sqrt(M**2.0 -1.0)))*180.0/pi)

def M2dnudM(G, M):
    '''Prandtl-Meyer relation: Mach number -> derivative of PM function'''

    # Outputs: dnu/dM [derivative of Prandtl-Meyer function wrt Mach number]
    # inputs:  G [specific heat ratio]
    #          M [Mach number]
    return (((G-1.0)*sqrt((G+1.0)/(G-1.0))*M)/((G+1.0)*sqrt((G-1.0)*(M**2.0-1.0)/(G+1.0))*
        ((G-1.0)*(M**2.0-1.0)/(G+1.0)+1.0)) - 1.0/(M*sqrt(M**2.0-1.0)))*180.0/pi

def nu2M(G, nu):
    '''Prandtl-Meyer relation: PM function -> Mach number'''

    # Outputs: M  [Mach number]
    # inputs:  G  [specific heat ratio]
    #          nu [Prandtl-Meyer function]

    # Newton-Rhapson Method
    M = [0.0, 1.5]
    while abs(M[0] - M[1])/M[1] > 1.0e-15:
        M[0] = M[1]
        M[1] = M[0] - (M2nu(G,M[0])-nu)/M2dnudM(G, M[0])
    return M[1]

def nu2mu(G, nu):
    '''Prandtl-Meyer relation: PM function -> Mach angle'''

    # Outputs: mu [Mach angle] in degrees
    # inputs:  G  [specific heat ratio]
    #          nu [Prandtl-Meyer function]
    M = nu2M(G,nu)
    return asin(1.0/M)*180.0/pi
