import numpy
import pylab
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def makeData ():
    x = numpy.arange (0, 1, 0.1)
    y = numpy.arange (0, 1, 0.1)
    xgrid, ygrid = numpy.meshgrid(x, y)

    #zgrid = numpy.sin (xgrid) * numpy.sin (ygrid) / (xgrid * ygrid)
    zgrid = (xgrid - 0.5)**2 + (ygrid - 0.5)**2
    return xgrid, ygrid, zgrid

x, y, z = makeData()

fig = pylab.figure()
axes = Axes3D(fig)

#axes.plot_surface(x, y, z, label = "y(s)")
axes.scatter(x,y)
axes.scatter(x,y,1)
axes.set_xlabel("x")                              # подпись у горизонтальной оси х
axes.set_ylabel("t")
axes.set_zlabel("y")
axes.legend()
pylab.show()

fig.savefig('1.png')