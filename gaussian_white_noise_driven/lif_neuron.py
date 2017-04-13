"""
analytics for leaky integrate-and-fire neurons driven by Gaussian white noise
"""

from analytics.decorators.cache_dec import cached
from analytics.decorators.param_dec import dictparams
from analytics.helpers import integrate, heav
from numpy import exp, cos, sin, sqrt, real, arctan, arctanh, log, abs, inf, pi
from math import erfc
from analytics.specfunc import pcfd

@dictparams
@cached
def r0(mu, D, tr, vr, vt):
    return 1./(tr + sqrt(pi) * integrate(lambda z: exp(z**2) * erfc(z), (mu-vt)/sqrt(2*D), (mu-vr)/sqrt(2*D)))


@dictparams
@cached
def powspec(fs, mu, D, tr, vr, vt):
    """Return the power spectrum of a GWN-driven LIF
    """
    delta = (vr**2-vt**2+2*mu*(vt-vr))/(4*D)
    io = 1j*(2*pi*fs)
    pcfdvr = pcfd(io,(mu-vr)/sqrt(D))
    pcfdvt = pcfd(io,(mu-vt)/sqrt(D))
    return r0(locals()) *  (abs(pcfdvt)**2 - exp(2*delta) * abs(pcfdvr)**2)/ abs(pcfdvt - exp(delta) * exp(io*tr) * pcfdvr)**2


@dictparams
@cached
def suscep(fs, mu, D, vr, vt, tr):
    """Return the 
    """
    io = 1j*2*pi*fs
    delta = (vr**2-vt**2+2*mu*(vt-vr))/(4*D)
    return r0(locals()) * io/sqrt(D)/(io-1) * (pcfd(io-1,(mu-vt)/sqrt(D)) - exp(delta) * pcfd(io-1,(mu-vr)/sqrt(D))) / (pcfd(io,(mu-vt)/sqrt(D)) - exp(delta + io*tr) * pcfd(io,(mu-vr)/sqrt(D)))

@dictparams
@cached
def suscep_noisemod(fs, mu, D, vr, vt, tr):
    io = 1j*2*pi*fs
    delta = (vr**2-vt**2+2*mu*(vt-vr))/(4*D)
    res = []
    return r0(locals()) * io*(io-1.)/D/(2-io) * (pcfd(io-2, (mu-vt)/sqrt(D)) - exp(delta) * pcfd(io-2, (mu-vr)/sqrt(D))) / (pcfd(io, (mu-vt)/sqrt(D)) - exp(delta + io*tr) * pcfd(io, (mu-vr)/sqrt(D)))
