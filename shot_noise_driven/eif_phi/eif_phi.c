#include <math.h>
#include <stdio.h>

/*
 * to compile as a libraryd
 *
 *      cc -fPIC -c -O2 eif_phi.c -o eif_phi.o
 *      cc --shared eif_phi.o -o libeif_phi.so 
 *
 */
 
double phi_integrand(const int n, const double* const args) /* v, mu, d, vtb */
{
    return 1./(args[1] - args[0] + args[2] * exp((args[0]-args[3])/args[2])); /* 1./(mu - v + d * exp((v-vtb)/d) */
}
