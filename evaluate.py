import numpy as np
from math import log10

def Polinomial_min_square(x, y):
    A_sum = np.zeros((4, 1))
    A_matrix = np.zeros((3,3))
    s = np.shape(x)
    for i in range(0, s[0]):
        for j in range(1, 5):
            A_sum[j-1] += x[i]**j
    A_matrix[0][0] = s[0]
    A_matrix[0][1] = A_sum[0]
    A_matrix[0][2] = A_sum[1]
    A_matrix[1][0] = A_sum[0]
    A_matrix[1][1] = A_sum[1]
    A_matrix[1][2] = A_sum[2]
    A_matrix[2][0] = A_sum[1]
    A_matrix[2][1] = A_sum[2]
    A_matrix[2][2] = A_sum[3]
    print(A_matrix)
    B_matrix = np.zeros((3, 1))
    for i in range(0, s[0]):
        for j in range(0, 3):
            B_matrix[j] += y[i] * (x[i]**j)
    print(B_matrix)
    A_inv = np.linalg.inv(A_matrix)
    print(A_inv)
    C = A_inv.dot(B_matrix)
    print(C)
    poly_y = np.zeros(s)
    for i in range(0, s[0]):
        poly_y[i] = C[0] + C[1]*x[i] + C[2]*(x[i]**2)
    return poly_y

def Hirst(array):
    X = np.zeros((5,))
    Y = np.zeros((5,))
    step = int(array.shape[0]/5)
    for i in range(0, 5):
        N = (1000 / 5) * (i+1)
        this_arr = array[0:(step-1)*(i+1)]
        R = np.max(this_arr) - np.min(this_arr)
        S = np.std(this_arr)
        Y[i] = log10(R/S)
        X[i] = log10(N)
    return [X, Y]