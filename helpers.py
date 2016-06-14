from scipy.integrate import quad

def heav(x):
    return x > 0

def integrate(f, a, b):
    return quad(f, a, b)[0]
