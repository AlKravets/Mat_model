import numpy as np
import pylab
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


a11 = np.ones((2,2))
a12 = 2*np.ones((2,2))
a21 = np.ones((2,2))*3
a22 = np.ones((2,2))*4

a1 = np.append(a11,a12, axis= 1)
a2 = np.append(a21,a22, axis =1)
a = np.append(a1,a2,axis =0)

print(a)