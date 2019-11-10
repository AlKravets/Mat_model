import math
import numpy as np




def G(s):
    if s[1] <= 0:
        return 0
    else:
        return 1/(4*math.pi*math.e**2*s[1])**0.5 *math.exp((-1*s[0]**2)/(4*math.e*s[1]))


def y(s: np.ndarray)-> float:
    res = 0
    #for i in range(s.shape[0]):
    for i in range(2):
        res+= (s[i] -0.5)**2
    #res = s[1]**2 + s[0]**3
    return res


def U(s):
    return 2*(s[1]-0.5)- 2*math.e**2
    # return 2*s[1] - 6*math.e**2*s[0]


if __name__ == '__main__':
    print(U([0,0]))