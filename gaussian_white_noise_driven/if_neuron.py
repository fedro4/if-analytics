"""
analytics for general integrate-and-fire neurons driven by Gaussian white noise
"""

from analytics.decorators.cache_dec import cached
from analytics.decorators.param_dec import dictparams
from analytics.helpers import integrate, heav
from numpy import exp, cos, sin, sqrt, real, arctan, arctanh, log, abs, inf
import mpmath as mp

infty=15

class PIF:
    def U(self, mu, v):
        return -mu*v
    def __hash__(self): # needed for caching
        return "PIFgwn"

class LIF:
    def U(self, mu, v):
        return 1./2 * (v-mu)**2
    def __hash__(self): # needed for caching
        return "LIFgwn"

class QIF:
    def U(self, mu, v):
        return -v**3/3 - mu*v
    def __hash__(self): # needed for caching
        return "QIFgwn"


@dictparams
@cached
def T1(model, mu, D, vr, vt, tr):
    """Return the first moment of the ISI density (i.e. the inverse firing rate)"""
    U = lambda x: model.U(mu,x)
    return 1./D * integrate(lambda x: integrate(lambda y: exp((U(x)-U(y))/D), -infty, x), vr, vt) + tr
    #return 1./D * float(mp.quad(lambda x: mp.quad(lambda y: mp.exp(U(x)-U(y)/D), (-infty, x)), (vr, vt))) + tr

#@dictparams
#@cached
#def dT2(model, mu, D, vr, vt, tr):   

@dictparams
@cached
def r0(model, mu, D, vr, vt, tr):
    """Return the firing rate"""
    return 1./T1(locals())

@dictparams
@cached
def P0(model, v, mu, D, vr, vt, tr, r0):
    """Return the stationary density"""
    U = lambda x: model.U(mu,x)
    #return 0. if v > vt else r0/D * integrate(lambda y: exp((U(y)-U(v))/D), vr if v < vr else v, vt)
    return 0. if v > vt else r0/D * float(mp.quad(lambda y: mp.exp((U(y)-U(v))/D), (vr if v < vr else v, vt)))
