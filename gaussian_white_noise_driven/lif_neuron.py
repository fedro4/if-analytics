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
def r0(fs, mu, D, tr, vr, vt):
    return 1./(tr + sqrt(pi) * integrate(lambda z: exp(z**2) * erfc(z), (mu-vt)/sqrt(2*D), (mu-vr)/sqrt(2*D)))


@dictparams
@cached
def powspec(fs, mu, D, tr, vr, vt):
    """Return the power spectrum of a GWN-driven LIF
    """
    def Del(mu, D):
        return (vr**2 - vt**2 + 2*mu*(vt-vr))/(4*D)

    io = 1j*(2*pi*fs)
    pcfdvr = pcfd(io,(mu-vr)/sqrt(D))
    pcfdvt = pcfd(io,(mu-vt)/sqrt(D))
    return r0(locals()) *  (abs(pcfdvt)**2 - exp(2 * Del(mu, D)) * abs(pcfdvr)**2)/ abs(pcfdvt - exp(Del(mu, D)) * exp(io*tr) * pcfdvr)**2


