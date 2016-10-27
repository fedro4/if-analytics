"""
analytics for leaky integrate-and-fire neurons driven by dichotomous noise
"""

from analytics.decorators.cache_dec import cached
from analytics.decorators.param_dec import dictparams
from numpy import exp, cos, sin, sqrt, real, arctan, arctanh, log, abs, pi
from analytics.helpers import integrate, heav
import analytics.dichotomous_noise_driven.if_neuron as ifana
from analytics.specfunc import hyp2f1
import numpy as np
import mpmath as mp


def z(mu, s, v):
    return (v-(mu-s))/2/s

@dictparams
@cached
def r0(mu, s, kp, km, vr, vt, tr):
    """Return the stationary firing rate"""
    return 1./ifana.T1(locals(), model=ifana.LIF())

@dictparams
@cached
def r0_altern(mu, s, kp, km, vr, vt, tr):
    """Alternative expression for the stationary firing rate"""
    #return ( 1./(tr + (kp+km) * integrate(lambda x: integrate(lambda y: abs(mu-y+s)**kp/abs(mu-x+s)**(kp+1) * abs(mu-y-s)**(km-1)/abs(mu-x-s)**km, x, mu-s), vr, vt) 
    return ( 1./(tr + (kp+km) * integrate(lambda x: integrate(lambda y: abs((mu-y+s)/(mu-x+s))**(kp) * abs((mu-y-s)/(mu-x-s))**(km) * 1./(mu-x+s) * 1./(mu-y-s), x, mu-s), vr, vt) 
           + (1-exp(-tr*(kp+km)))/(kp+km) * (-1+(kp+km) * integrate(lambda x: abs((mu-x+s)/(mu-vr+s))**kp * abs((mu-x-s)/(mu-vr-s))**km * 1./(mu-x-s), vr, mu-s)) ))


@dictparams
@cached
def powspec(fs, mu, s, kp, km, vr, vt, tr):
    """Return the power spectrum"""
    return r0(locals()) * (1.+2*st_rate(locals()).real)

@dictparams
@cached
def st_rate(fs, mu, s, kp, km, vr, vt, tr):
    """Return the spike-triggered rate"""
    zr = z(mu, s, vr)
    zt = z(mu, s, vt)
    io = 1j*(2*pi*fs)
    
    F = lambda u: (1.-u)**(io-kp) * hyp2f1(km,-kp,km-io,u)
    G = lambda u: (1.-u)**(1+io-kp) * hyp2f1(1.+km,1.-kp,1+km-io,u)
    
    etr = exp(-(kp+km)*tr)
    return 1./(((km-io)*(kp+km)*exp(-io*tr)*F(zt))/((km-io)*(kp*etr+km)*F(zr)+km*kp*(1.-etr)*G(zr)) - 1)

@dictparams
@cached
def powspec_highom(fs, mu, s, kp, km, vr, vt, tr):
    """Return the high-frequency behavior of the power spectrum"""
    o = 2*pi*fs
    Td = np.log((mu + s - vr) / (mu + s - vt)) + tr
    Ppp = (kp*exp(-(kp+km)*tr)+km)/(kp+km)
    return r0(locals()) * (1.-Ppp*Ppp*np.exp(-2*kp*(Td-tr)))/(1+Ppp*Ppp*np.exp(-2*kp*(Td-tr))-2*Ppp*np.exp(-kp*(Td-tr))*np.cos(o*Td))

@dictparams
#@cached # cached seems a bad idea here due to the hash(-1) == hash(-2) issue with ints
def powspec_disc_n(n, fs, mu, s, kp, km, vr, vt, tr):
    """Return the n'th Lorentzian and its width"""
    Td = ifana.LIF().Tdp(mu, s, vr, vt) + tr
    Ppp = (kp*exp(-(kp+km)*tr)+km)/(kp+km)
    kpbar = (kp*(Td-tr)-log(Ppp))/Td
    return 1./Td * 2*kpbar/(kpbar**2 + (2*pi*(fs - n*1./Td))**2), kpbar

@dictparams
@cached
def powspec_exp(fs, mu, s, kp, km, vr, vt, tr):
    zr = z(mu, s, vr)
    zt = z(mu, s, vt)
    io = 1j*(2*pi*fs)
    H = lambda u: u**(1+io-km) * hyp2f1(1-km,1+kp,2-km+io,u)
    return r0(locals()) * (np.abs(H(zt))**2-np.abs(H(zr))**2)/np.abs(H(zt)-H(zr))**2   


@dictparams
@cached
def powspec_disc(fs, mu, s, kp, km, vr, vt, tr):
    """Return the part of the power spectrum (a superposition of Lorentzians) that is due to the delta-peaks in the spike-triggered rate""" 
    o=2*pi*fs
    Td = ifana.LIF().Tdp(mu, s, vr, vt) + tr
    nmax = int(10000*np.max(fs)/(1./Td))
    nmin = -nmax 
    print nmin, nmax
    res = np.zeros(len(fs));
    for n in range(nmin, nmax, 1):
        res += powspec_disc_n(locals(), n=n)[0]
    return r0(locals()) * (res)

@dictparams
@cached
def suscep(fs, mu, s, kp, km, vr, vt, tr):
    """Return the susceptibility with respect to a current modulation"""
    zr = z(mu, s, vr)
    zt = z(mu, s, vt)
    io = 1j*(2*pi*fs)
    F = lambda u: (1.-u)**(io-kp) * hyp2f1(km,-kp,km-io,u)
    G = lambda u: (1.-u)**(1.+io-kp) * hyp2f1(1.+km,1.-kp,1+km-io,u)
    Fp = lambda u: -io*(kp+km-io)/(km-io) * (1.-u)**(io-kp-1) * hyp2f1(km,-kp,1.+km-io,u)
    Gp = lambda u: -io*(kp+km-io)/(1.+km-io) * (1.-u)**(io-kp) * hyp2f1(1.+km,1.-kp,2+km-io,u)
    etr = exp(-(kp+km)*tr)
    return -r0(locals())/(2.*s) * 1./(io-1) * ((km-io) * ((kp*etr+km) * Fp(zr) - (kp+km)*Fp(zt)) +km*kp*(1.-etr) * Gp(zr) )/( (km-io) * ((kp*etr+km) * exp(io*tr)*F(zr) - (kp+km)*F(zt)) +km*kp*(1.-etr) *  exp(io*tr)*G(zr)  )

@dictparams
@cached
def suscep_highom(fs, mu, s, kp, km, vr, vt, tr):
    """Return the high-frequency behavior of the susceptibility with respect to a current modulation"""
    o = 2*pi*fs
    zr = z(mu, s, vr)
    zt = z(mu, s, vt)
    Td = np.log((mu + s - vr) / (mu + s - vt)) + tr
    Ppp = (kp*exp(-(kp+km)*tr)+km)/(kp+km)
    return r0(locals())/(2*s) * (1.-Ppp*exp(-(kp+1)*(Td-tr))*exp(1j*o*(Td-tr)))/((1-zt)*(1.-Ppp*exp(-kp*(Td-tr))*exp(1j*o*Td)))

