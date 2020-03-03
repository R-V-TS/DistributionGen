import numpy as np
from random import uniform
from math import log, pow

def parito_gen(alfa, k, num_of_sample):
    a = np.zeros((num_of_sample,1))

    for i in range(0, num_of_sample):
        r = uniform(0, 1)
        a[i] = k/(r**(1/alfa))
    return a

def weibula_gen(alfa, k, num_of_sample):
    a = np.zeros((num_of_sample,1))

    for i in range(0, num_of_sample):
        r = uniform(0, 1)
        a[i] = k * pow(-log(r), (1/alfa))
    return a
