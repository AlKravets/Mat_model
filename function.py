import math
import numpy as np




def G(s):
    if s[1] <= 0:
        return 0.
    else:
        return 1/(4*math.pi*(math.e**2)*s[1])**0.5 *math.exp((-1*s[0]**2)/(4*math.e*s[1]))

        # return 1/(2*(math.pi * s[1])**0.5) * math.exp(-( s[1] + (s[0]-s[1])**2 / (4* s[1] )))  # пример из методички


def y(s: np.ndarray)-> float:
    res = 0
    #for i in range(s.shape[0]):
    # for i in range(2):
    #     res+= (s[i] -0.5)**2
    res = 10*s[1]**2 + s[0]**3

    # res = s[0] + 1 + s[1]/4       # пример из методички
    return res


def U(s):
    # return 2*(s[1]-0.5)- 2*math.e**2
    return 20*s[1] - 6*math.e**2*s[0]
    # return  9/4 + s[1]/4 + s[0]  # пример из методички


if __name__ == '__main__':
    print(U([0,0]))