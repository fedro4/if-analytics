"""
analytics for quadratic integrate-and-fire neurons driven by Gaussian white noise
"""

from analytics.decorators.cache_dec import cached
from analytics.decorators.param_dec import dictparams
from analytics.helpers import integrate, heav
from numpy import exp, cos, sin, sqrt, real, arctan, arctanh, log, abs, inf, pi

@dictparams
@cached
def T1(mu, D, tr):
    """Return the first moment of the ISI density.
    This single-integral form is due to Brunel & Latham, Neural Comp. 2003
    It assumes vr, vt -> +- infty"""
    return sqrt(pi) * integrate(lambda z: exp(-mu * z**2 - D**2/12 * z**6), -inf, inf) + tr

@dictparams
@cached
def r0(model, mu, D, tr):
    """Return the firing rate"""
    return 1./T1(locals())
