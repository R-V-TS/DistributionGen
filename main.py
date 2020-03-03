import numpy as np
from scipy.stats import pareto
from matplotlib import pyplot as plt
from generators import parito_gen, weibula_gen
from evaluate import Hirst
from statsmodels.tsa.stattools import acf
from scipy.special import gamma
from math import exp

def D_calc(alpha, k):
    return (alpha*(k**2))/(((alpha-1)**2)*(alpha - 2))

def D_W_calc(alpha, k):
    return pow(k, 2)*(gamma(1+(2/alpha)) - (gamma(1+(1/alpha))**2))


M = float(input("Введите значение мат. ожидания: "))
D = float(input("Введите значение дисперсии: "))
E = float(input("Введите точность подбора значений: "))
mode = int(input("Введите номер расспределения (1 - Парето, 2 - Вейбула): "))

if 0 > mode > 2:
    print("1 или 2 нужно вводить!")
else:
    if mode == 1:
        alpha = 0.000001
        k = (M * (alpha - 1)) / alpha

        while abs(D - D_calc(alpha, k)) > E:
            alpha += 0.001
            k = (M * (alpha - 1)) / alpha

        D = D_calc(alpha, k)
        print("Матю ож. = " + str(M))
        print("Дисперсия = " + str(D))
        print("Aльфа = " + str(alpha))
        print("K = " + str(k))

        x = np.arange(alpha, 10, 0.1)
        # f = np.array([((alpha/k)*(((i)/k)**(-(alpha+1)))) for i in x])
        f = np.array([((k * (alpha ** k)) / (i ** (k + 1))) for i in x])

        parito = parito_gen(k, alpha, 1000)  # Generate Parito

        plt.figure()
        plt.hist(parito, bins=100, normed=True)
        plt.plot(x, f, 'r-')
        plt.show()

        [X, Y] = Hirst(parito)
        A = np.vstack([X, np.ones(X.shape[0])]).T
        b1, b2 = np.linalg.lstsq(A, Y)[0]

        plt.figure()
        plt.plot(X, Y, 'ro-')
        plt.plot(X, b1 * X + b2, 'b')
        plt.show()

        print("B2 = " + str(b1))

        ACF_parito = acf(parito)
        print("ACF coff's: ")
        print(ACF_parito)
        plt.figure()
        plt.plot(range(0, np.shape(ACF_parito)[0]), ACF_parito, '-')
        plt.show()

    else:
        alpha = 1.000001
        k = M/gamma(1+(1/alpha))
        while abs(D - D_W_calc(alpha, k)) > E:
            alpha += 0.01
            k = M/gamma(1+(1/alpha))

        D = D_W_calc(alpha, k)
        print("Матю ож. = " + str(M))
        print("Дисперсия = " + str(D))
        print("Aльфа = " + str(alpha))
        print("K = " + str(k))

        weibula = weibula_gen(alpha, k, 1000)
        x = np.arange(np.min(weibula), np.max(weibula) + 1, 0.1)
        #f = np.array([((alpha/k)*pow((i/k), (alpha-1)))*exp(pow(-(i/k), alpha)) for i in x], dtype=float)
        f = np.array([((alpha/k)*(((i)/k)**(alpha-1)))*exp(-(i/k)**alpha) for i in x])
        plt.figure()
        plt.hist(weibula, bins=100, normed=True)
        plt.plot(x, f, 'r-')
        plt.show()

        [X, Y] = Hirst(weibula)
        A = np.vstack([X, np.ones(X.shape[0])]).T
        b1, b2 = np.linalg.lstsq(A, Y)[0]

        plt.figure()
        plt.plot(X, Y, 'ro-')
        plt.plot(X, b1*X + b2, 'b')
        plt.show()

        print("B2 = " + str(b1))

        ACF_w = acf(weibula)
        print("ACF coff's: ")
        print(ACF_w)
        plt.figure()
        plt.plot(range(0, np.shape(ACF_w)[0]), ACF_w, '-')
        plt.show()