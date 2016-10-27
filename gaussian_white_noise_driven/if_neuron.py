"""
analytics for general integrate-and-fire neurons driven by Gaussian white noise
"""

import qif_neuron as qifana
from analytics.decorators.cache_dec import cached
from analytics.decorators.param_dec import dictparams
from analytics.helpers import integrate, heav
from numpy import exp, cos, sin, sqrt, real, arctan, arctanh, log, abs, inf
import mpmath as mp
from scipy.integrate import quad
import math

epsrel=1e-3

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

class EIF:
    def __init__(self, d, vtb): # all model parameters should belong to the model, also mu for pif, qif etc.
        self.d = d
        self.vtb = vtb
    def U(self, mu, v):
        d = self.d
        vtb = self.vtb
        return 1./2*(mu-v)**2-d**2*math.exp((v-vtb)/d)
    def __hash__(self): # needed for caching
        return "EIFgwn"  + repr(self.__dict__)
    


@dictparams
@cached
def T1(model, mu, D, vr, vt, tr):
    """Return the first moment of the ISI density (i.e. the inverse firing rate)"""
    U = lambda x: model.U(mu,x)
    return 1./D * quad(lambda x: quad(lambda y: exp((U(x)-U(y))/D), -infty, x, epsrel=epsrel)[0], vr, vt, epsrel=epsrel)[0] + tr
    #return 1./D * float(mp.quad(lambda x: mp.quad(lambda y: mp.exp(U(x)-U(y)/D), (-infty, x)), (vr, vt))) + tr

#@dictparams
#@cached
#def dT2(model, mu, D, vr, vt, tr):   

@dictparams
@cached
def r0(model, mu, D, vr, vt, tr):
    """Return the firing rate"""
    if vr < -10 or vt > 10 and model.__hash__() == "QIFgwn":
        # the general theory works pretty badly for QIFs with vr->-infty, vt->infty. 
        return 1./qifana.T1(locals())
    return 1./T1(locals())

@dictparams
@cached
def P0(model, v, mu, D, vr, vt, tr):
    """Return the stationary density"""
    U = lambda x: model.U(mu,x)
    #return 0. if v > vt else r0/D * integrate(lambda y: exp((U(y)-U(v))/D), vr if v < vr else v, vt)
    return 0. if v > vt else r0(locals())/D * float(mp.quad(lambda y: mp.exp((U(y)-U(v))/D), (vr if v < vr else v, vt)))
