"""
analytics for leaky integrate-and-fire neurons driven by excitatory shot noise with exponentially distributed weights
"""

from analytics.decorators.cache_dec import cached
from analytics.decorators.param_dec import dictparams
from numpy import exp, cos, sin, sqrt, real, arctan, arctanh, log, abs, pi
from analytics.helpers import integrate, heav
import analytics.shot_noise_driven.if_neuron as ifana
from analytics.specfunc import hyp1f1
import numpy as np
import mpmath as mp

@dictparams
@cached
def r0(mu, rin_e, a_e, vr, vt, tr): 
    """Return the stationary firing rate of an LIF driven by excitatory shot noise. 
    Adapted from the expression derived in 
    Richardson, M. J. & Swarbrick, R. Phys. Rev. Lett., APS, 2010, 105, 178102"""

    T = tr+float(integrate(lambda c: 1./c*(1-a_e*c)**rin_e * (exp(c*(vt-mu))/(1-a_e*c)-exp(c*(vr-mu))), 0, 1./a_e).real)
    return 1./T

@dictparams
@cached
def powspec(fs, mu, rin_e, a_e, vr, vt, tr):
    """Return the power spectrum of an LIF driven by excitatory shot noise"""
    io = 1j*(2*pi*fs)
    Fsn = lambda v: hyp1f1(-io,rin_e-io,(v-mu)/a_e)
    Gsn = lambda v: hyp1f1(-io,1+rin_e-io,(v-mu)/a_e)
    r0sn = r0(locals())
    return r0sn * (abs(exp(-io*tr)*Fsn(vt))**2-abs(rin_e/(rin_e-io)*Gsn(vr))**2)/abs(exp(-io*tr)*Fsn(vt)-rin_e/(rin_e-io)*Gsn(vr))**2

@dictparams
@cached
def suscep(fs, mu, rin_e, a_e, vr, vt, tr):
    """Return the susecptibility of an LIF driven by excitatory shot noise with respect to a current modulation"""
    io = 1j*(2*pi*fs)
    Fsn = lambda v: hyp1f1(-io,rin_e-io,(v-mu)/a_e)
    Gsn = lambda v: hyp1f1(-io,1+rin_e-io,(v-mu)/a_e)
    Fsnp = lambda v: -io/(a_e*(rin_e-io)) * hyp1f1(1-io, 1+rin_e-io, (v-mu)/a_e)
    Gsnp = lambda v: -io/(a_e*(1+rin_e-io)) * hyp1f1(1-io, 2+rin_e-io, (v-mu)/a_e)
    r0sn = r0(locals())
    return -r0sn/(io-1) * (Fsnp(vt)-rin_e/(rin_e-io)*Gsnp(vr))/(Fsn(vt)-exp(io*tr)*rin_e/(rin_e-io)*Gsn(vr))

@dictparams
@cached
def suscep_highf(fs, mu, rin_e, a_e, vr, vt, tr):
    """Return the high-frequency limit of an LIF driven by excitatory shot noise with respect to a current modulation"""
    io = 1j*(2*pi*fs)
    return -1./io * r0(locals())/a_e


@dictparams
@cached
def suscep_ratemod(fs, mu, rin_e, a_e, vr, vt, tr):
    """Return the susceptibility of a exc-shot-noise-driven LIF with respect to a modulation of the input rate
    XXX check: I think as it is, this implementation only works for mu < vr""" 
    io = 1j*(2*pi*fs)
    dv = float(vt-vr)/1e3
    vs=np.arange(ifana.LIF().intervals(mu=mu, vr=vr, vt=vt)[0][0], vt, dv)
    Fsn = lambda v: hyp1f1(-io,rin_e-io,(v-mu)/a_e)
    Gsn = lambda v: hyp1f1(-io,1+rin_e-io,(v-mu)/a_e)
    prms = locals()
    P0 = lambda v: ifana.P0(prms, model=ifana.LIF(), v=v)
    
    return np.sum(rin_e * np.array([[P0(v) for o in io] for v in vs]) * ((rin_e-io)*np.array([Fsn(v) for v in vs]) -rin_e*np.array([Gsn(v) for v in vs])), axis=0)*dv / ((rin_e-io)*Fsn(vt)-rin_e*exp(io*tr)*Gsn(vr))

@dictparams
@cached
def r0_d(tau, mu, rin_e, rin_i, a_e, a_i, vr, vt, tr): 
    T = tr + tau * float(
        mp.quad(
            lambda c: 1./c*(1-a_e*c)**(rin_e*tau) * (1.0 + a_i*c)**(rin_i*tau) * (
                mp.exp(c*(vt-mu))/(1-a_e*c)-mp.exp(c*(vr-mu))), (0, 1./a_e)).real)
    return 1./T


@dictparams
@cached
def powspec_d(fs, tau, mu, rin_e, rin_i, a_e, a_i, vr, vt, tr):
    return  r0_d(locals()) * (1.+2.0*np.array([complex(st_rate_d(locals(), f=f)) for f in fs]).real)

@dictparams
@cached
def st_rate_d(f, tau, mu, rin_e, rin_i, a_e, a_i, vr, vt, tr):
    """a_i must be negative!"""
    io = - 1j * 2.0 * np.pi * f
    def inv_Z_0(s):
        return (1.0 - a_e*s)**(tau*rin_e) * (1.0 - a_i*s)**(tau*rin_i)
    def A(s):
        return tau*a_e*rin_e / (1.0 - a_e*s) + tau*a_i*rin_i / (1.0 - a_i*s)        
    def B(s):        
        return mp.exp(s*(vt-mu)) / (1.0 - a_e*s) - mp.exp(s*(vr-mu)-io*tr)
    def dB_ds(s):        
        return mp.exp(s*(vt-mu)) / (1.0 - a_e*s) * ((vt-mu) + a_e / (1.0 - a_e*s)) - (vr-mu)*mp.exp(s*(vr-mu)-io*tr)
    def f_num(s):        
        return s**(io * tau) * mp.exp(s*(vr-mu)) * mp.exp(-io*tr) * inv_Z_0(s) * (A(s) - (vr-mu))
    def f_den(s):
        return s**(io * tau) * inv_Z_0(s) * (A(s) * B(s) - dB_ds(s))    
    f_num = np.frompyfunc(f_num,1,1)
    f_den = np.frompyfunc(f_den,1,1)
    I_num = mp.quad(f_num, (0.00,1./a_e),maxdegree=40)
    I_den = mp.quad(f_den, (0.00,1./a_e),maxdegree=40)
    return I_num / I_den

