import numpy as np

# These functions are from the original approach

# Definitions for AMPCA and GMPCA

def AMPCA(proba, y):
    sum = 0
    i = 0
    for sel_mode in y:
        sum = sum + proba[i, sel_mode]
        i += 1
    N = i-1
    if N == 0: return 0
    return sum/N

def CEL(proba, y):
    sum = 0
    i = 0
    for sel_mode in y:
        sum = sum + np.log(proba[i, sel_mode])
        i += 1
    N = i-1
    return -sum/N

def GMPCA(proba, y):
    return np.exp(-CEL(proba, y))