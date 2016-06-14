"""
analytics for general integrate-and-fire neurons driven by excitatory shot noise with exponentially distributed weights
"""


from analytics.decorators.cache_dec import cached
from analytics.decorators.param_dec import dictparams
from analytics.helpers import integrate, heav
from numpy import exp, cos, sin, sqrt, real, arctan, arctanh, log, abs

dv = 0.# 1e-10

infty=15

class PIF:
    def f(self, v, mu):
        return mu
    def fp(self, v, mu):
        return 0.
    def phi(self, v, mu, rin_e, a_e):
        return  v/a_e + rin_e * v/mu
    @cached
    def intervals(self, mu, vr, vt):
        return ( ((-infty, vt, vt+dv),) if mu<0 else ((vr, vt, vr-dv),) )
    def __hash__(self): # needed for caching
        return "PIFsn"

class LIF:
    def f(self, v, mu):
        return mu - v
    def fp(self, v, mu):
        return -1.
    def phi(self, v, mu, rin_e, a_e):
        return v/a_e - rin_e * log(abs(mu-v))
    @cached
    def intervals(self, mu, vr, vt):
        sfp = mu
        if sfp < vr:
            return (sfp, vt, vt+dv),
        elif sfp > vt:
            return (vr, vt, vr-dv),
        else:
            return ((vr, sfp, vr-dv), (sfp, vt, vt+dv))
    def __hash__(self): # needed for caching
        return "LIFsn"

class QIF:
    def f(self, v, mu):
        return mu + v**2
    def fp(self, v, mu):
        return 2*v
    def phi(self, v, mu, rin_e, a_e):
        return  v/a_e + rin_e * (1./sqrt(mu)*arctan(v/sqrt(mu)) if mu > 0 else 
                    -1./sqrt(-mu) * (arctanh(v/sqrt(-mu)) if (sqrt(-mu) > v and v > -sqrt(-mu)) 
                                        else .5*log((v/sqrt(-mu)+1.)/(v/sqrt(-mu)-1.))))
    @cached
    def intervals(self, mu, vr, vt):
        if mu>0: # we do not have fps
            return (vr, vt, vr),
        else:
            sfp = -sqrt(-mu)
            ufp = sqrt(-mu)
            if self.f(vr, mu)>0:
                if vr < sfp < vt:
                    if sfp < ufp < vt:
                        return ((vr, sfp, vr-dv), (sfp, ufp, ufp), (ufp, vt, ufp))
                    else:
                        return ((vr, sfp, vr-dv), (sfp, vt, vt+dv))
                else:
                    return (vr, vt, vr-dv),
            else:
                if vr < ufp < vt:
                    return ((sfp, ufp, ufp), (ufp, vt, ufp))
                else:
                    return (sfp, vt, vt+dv),
    def __hash__(self): # needed for caching
        return "QIFsn"


@dictparams
@cached
def alpha(model, mu, rin_e, a_e, vr, vt, tr):
    """Return the fraction of trajectories that cross due to an incoming spike (i.e. not by drifting)"""
    phi = lambda v: model.phi(v, mu, rin_e, a_e)
    f = lambda v: model.f(v, mu)
    ln, rn, cn = model.intervals(mu, vr, vt)[-1]
    return 1. - (heav(vr-cn)*exp(phi(vr)-phi(vt))+1./a_e*integrate(lambda x: exp(phi(x)-phi(vt))*heav(x-vr), cn, vt))


@dictparams
@cached
def T1(model, mu, rin_e, a_e, vr, vt, tr):
    """Return the first moment of the ISI density (i.e. the inverse firing rate)"""
    phi = lambda v: model.phi(v, mu, rin_e, a_e)
    f = lambda v: model.f(v, mu)
    
    def plusint(l, r):
        return integrate(lambda x: exp(phi(x)-phi(r))/f(x), l, r)

    def minusint(l, r):
        return integrate(lambda x: exp(-phi(x)+phi(l))/f(x), l, r)

    res = 0.
    ints = model.intervals(mu, vr, vt)
    for i in ints:
        l, r, c = i
        cbar = l if c == r else r
        res += 1./a_e * integrate(lambda x: heav(x-vr) * minusint(x, cbar), l, r)

    ln, rn, cn = ints[-1]
    l0, r0, c0 = ints[0]
    c0bar = l0 if c0 == r0 else r0
   
    return res + tr + minusint(vr, c0bar)

@dictparams
@cached
def r0(model, mu, rin_e, a_e, vr, vt, tr):
    """Return the firing rate"""
    return 1./T1(locals())

@dictparams
@cached
def P0(model, v, mu, rin_e, a_e, vr, vt, tr):
    """Return the stationary density at a particular voltage
    v can *not* be an array!
    """
    phi = lambda v: model.phi(v, mu, rin_e, a_e)
    f = lambda v: model.f(v, mu)
    fp = lambda v: model.fp(v, mu)
    ints = model.intervals(mu, vr, vt)
    if v < ints[0][0] or v > ints[-1][1]:
        return 0.

    r0 = 1./T1(locals())

    # which interval are we in
    if v <= ints[0][0]:
        return 0.
    l, r, c = (None, None, None)
    for l, r, c in ints:
        if l < v < r:
            break

    return r0/f(v) * (  ((1.if c>vr else 0.)-heav(v-vr)) * (-exp(phi(vr)-phi(v)))
                                +1./a_e * integrate(lambda x: exp(phi(x)-phi(v))*(heav(x-vr)-heav(x-vt)), c, v))

