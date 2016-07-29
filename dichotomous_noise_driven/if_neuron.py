"""
analytics for general integrate-and-fire neurons driven by dichotomous noise
"""

from analytics.decorators.cache_dec import cached
from analytics.decorators.param_dec import dictparams
from analytics.helpers import integrate, heav
from numpy import exp, cos, sin, sqrt, real, arctan, arctanh, log, abs

infty=15

dv = 0.# 1e-10

# the following classes describe models (PIF, LIF, QIF ...) for use with dichotomous noise

class PIF:
    def f(self, v, mu):
        return mu
    def fp(self, v, mu):
        return 0.
    def phi(self, v, mu, s, kp, km):
        return  v*(kp/(mu + s) + km/(mu - s))
    def Tdp(self, mu, s, vr, vt): # deterministic ISI in + state without tr!
        return (vt-vr)/(mu+s)+tr
    def Tdm(self, mu, s, vr, vt):
        return (vt-vr)/(mu-s)+tr
    @cached
    def intervals(self, mu, s, vr, vt):
        return ( ((-infty, vt, vt+dv),) if mu-s<0 else ((vr, vt, vr-dv),) )
    def __hash__(self): # needed for caching
        return "PIF"

class LIF:
    def f(self, v, mu):
        return mu - v
    def fp(self, v, mu):
        return -1.
    def phi(self, v, mu, s, kp, km):
        return -kp * log(abs(mu+s-v)) - km*log(abs(mu-s-v))
    def Tdp(self, mu, s, vr, vt): # deterministic ISI in + state without tr!
        return log((mu+s-vr)/(mu+s-vt))
    def Tdm(self, mu, s, vr, vt):
        return log((mu-s-vr)/(mu-s-vt))
    @cached
    def intervals(self, mu, s, vr, vt):
        sfp = mu-s
        if sfp < vr:
            return (sfp, vt, vt+dv),
        elif sfp > vt:
            return (vr, vt, vr-dv),
        else:
            return ((vr, sfp, vr-dv), (sfp, vt, vt+dv))
    def __hash__(self): # needed for caching
        return "LIF"

class QIF:
    def f(self, v, mu):
        return mu + v**2
    def fp(self, v, mu):
        return 2*v
    
    def phi(self, v, mu, s, kp, km):
        return  (kp/sqrt(mu+s) * arctan(v/sqrt(mu+s)) + km * 
                (1./sqrt(mu-s)*arctan(v/sqrt(mu-s)) if mu > s else 
                    -1./sqrt(s-mu) * (arctanh(v/sqrt(s-mu)) if (sqrt(s-mu) > v and v > -sqrt(s-mu)) 
                                        else .5*log((v/sqrt(s-mu)+1.)/(v/sqrt(s-mu)-1.)))))
    def Tdp(self, mu, s, vr, vt): # deterministic ISI in + state without tr
        return (arctan(vt/sqrt(mu+s))-arctan(vr/sqrt(mu+s)))/sqrt(mu+s) 
    def Tdm(self, mu, s, vr, vt):
        return (arctan(vt/sqrt(mu-s))-arctan(vr/sqrt(mu+s)))/sqrt(mu-s)
    @cached
    def intervals(self, mu, s, vr, vt):
        if s<mu: # we do not have fps
            return (vr, vt, vr),
        else:
            sfp = -sqrt(s-mu)
            ufp = sqrt(s-mu)
            if self.f(vr, mu)-s>0:
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
        return "QIF"

@dictparams
@cached
def if_dicho_alpha(model, mu, s, kp, km, vr, vt, tr):
    """Return the fraction of trajectories crossing the threshold in the + state"""
    phi = lambda v: model.phi(v, mu, s, kp, km)
    f = lambda v: model.f(v, mu)
    ekt = exp(-(kp+km)*tr)
    
    if f(vt)-s < 0:
        return 1.
    else:
        ln, rn, cn = model.intervals(mu, s, vr, vt)[-1]
        return 1.- (heav(vr-cn) * kp/(kp+km) * exp(phi(vr)) * (1.-ekt) + kp * integrate(lambda x: exp(phi(x))*heav(x-vr)/(f(x)+s), cn, vt)) / (exp(phi(vt))-heav(vr-cn)*ekt*exp(phi(vr)))

@dictparams
@cached
def T1(model, mu, s, kp, km, vr, vt, tr):
    """Return the mean first passage time"""
    phi = lambda v: model.phi(v, mu, s, kp, km)
    f = lambda v: model.f(v, mu)
    ekt = exp(-(kp+km)*tr)
    
    def plusint(l, r):
        return integrate(lambda x: exp(phi(x)-phi(r))/(f(x)+s), l, r)

    def minusint(l, r):
        return integrate(lambda x: exp(-phi(x)+phi(l))/(f(x)-s), l, r)

    al = if_dicho_alpha(locals())

    res = 0.
    ints = model.intervals(mu, s, vr, vt)
    for i in ints:
        l, r, c = i
        cbar = l if c == r else r
        res += (kp+km) * integrate(lambda x: heav(x-vr)/(f(x)+s) * minusint(x, cbar), l, r)

    ln, rn, cn = ints[-1]
    l0, r0, c0 = ints[0]
    c0bar = l0 if c0 == r0 else r0
   
    return res + tr + plusint(cn, vt) + ((1-al)/kp*ekt+1./(kp+km)*(1-ekt)) * (exp(phi(vr)-phi(c0bar))-1.+(kp+km) * minusint(vr, c0bar))
   
@dictparams
@cached
def P0(model, v, mu, s, kp, km, vr, vt, tr):
    """Return the stationary density. v may not be an array!"""
    phi = lambda v: model.phi(v, mu, s, kp, km)
    f = lambda v: model.f(v, mu)
    fp = lambda v: model.fp(v, mu)
    ekt = exp(-(kp+km)*tr)
    ints = model.intervals(mu, s, vr, vt)
    if v < ints[0][0] or v > ints[-1][1]:
        return 0.

    r0 = 1./if_dicho_T1(model, mu, s, kp, km, vr, vt, tr)
    al = if_dicho_alpha(model, mu, s, kp, km, vr, vt, tr)

    Gd = (2*al-1)*ekt + (km-kp)/(kp+km) * (1.-ekt)

    # which interval are we in
    if v <= ints[0][0]:
        return 0.
    l, r, c = (None, None, None)
    for l, r, c in ints:
        if l < v < r:
            break

    return r0/(f(v)**2-s**2) * (
            # (heav(c-vr)-heav(v-vr)) * (s*Gd-f(vr))*exp(phi(vr)-phi(v))
            #-(heav(c-vt)-heav(v-vt)) * (s*(2*al-1)-f(vt))*exp(phi(vt)-phi(v))
              ((1.if c>vr else 0.)-heav(v-vr)) * (s*Gd-f(vr))*exp(phi(vr)-phi(v))
             -((1.if c>=vt else 0.)-heav(v-vt)) * (s*(2*al-1)-f(vt))*exp(phi(vt)-phi(v))
            + integrate(lambda x: exp(phi(x)-phi(v))*(heav(x-vr)-heav(x-vt))*(fp(x)+kp+km), c, v))

