import math
import numpy as np
import pylab
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import function as fn
import constant as cn

def show_constant():
    fig, axes = plt.subplots(figsize=(12,4))  

    axes.scatter(cn.s_0[:,0],cn.s_0[:,1],s =0.1) # начальные наблюдения
    axes.scatter(cn.s_g[:,0],cn.s_g[:,1], s=0.1) # граничные наблюдения
    axes.scatter(cn.s_m[:,0],cn.s_m[:,1], s=5, marker="x") # возбуждения
    axes.scatter(cn.s_m_0[:,0],cn.s_m_0[:,1], s=5, marker="o") # начальные возбуждения
    axes.scatter(cn.s_m_g[:,0],cn.s_m_g[:,1], s=5) # граничные возбуждения
    axes.set_xlabel("x")                              
    axes.set_ylabel("t")
    plt.legend(("s_0", "s_g", "s_m", "s_m_0", "s_m_g"))
    fig.savefig('1_0.png')

def show_test_y():
    x = np.arange (0, 1, 0.1)
    y = np.arange (0, 1, 0.1)
    xgrid, ygrid = np.meshgrid(x, y)

    zgrid = (xgrid - 0.5)**2 + (ygrid - 0.5)**2

    fig1 = pylab.figure()
    axes = Axes3D(fig1)

    axes.plot_surface(xgrid,ygrid,zgrid)
    axes.set_xlabel("x")                              # подпись у горизонтальной оси х
    axes.set_ylabel("t")
    axes.set_zlabel("y")
    fig1.savefig('1_1.png')

def y_model(s, u, s_u):
    res =0
    for i in range(u.shape[0]):
        res += fn.G(s-s_u[i])*u[i]
    return res


def create_slar():
    u_ = np.zeros(cn.M_0+cn.M_g)

    

    Y_0_new = np.array([cn.Y_0[i] - y_model(cn.s_0[i], cn.u, cn.s_m) for i in range(cn.R_0)])
    Y_g_new = np.array([cn.Y_g[i] - y_model(cn.s_g[i], cn.u, cn.s_m) for i in range(cn.R_g)])

    Y_ = np.append(Y_0_new, Y_g_new)

    A_11 = np.zeros((cn.R_0, cn.M_0))
    for i in range(cn.R_0):
        for j in range(cn.M_0):    
            A_11[i][j] = fn.G(cn.s_0[i] - cn.s_m_0[j])

    A_12 = np.zeros((cn.R_0, cn.M_g))
    for i in range(cn.R_0):
        for j in range(cn.M_g):
            A_12[i][j] = fn.G(cn.s_0[i] - cn.s_m_g[j])
        
    A_21 = np.zeros((cn.R_g, cn.M_0))
    for i in range(cn.R_g):
        for j in range(cn.M_0):
            A_21[i][j] = fn.G(cn.s_g[i] - cn.s_m_0[j])
    
    A_22 = np.zeros((cn.R_g, cn.M_g))
    for i in range(cn.R_g):
        for j in range(cn.M_g):
            A_22[i][j] = fn.G(cn.s_g[i] - cn.s_m_g[j])
    
    A_1 = np.append(A_11, A_12, axis= 1)
    A_2 = np.append(A_21, A_22, axis= 1)
    
    A = np.append(A_1, A_2, axis= 0)

    A_plus = np.dot(np.transpose(A), np.linalg.inv(np.dot(A,np.transpose(A))))

    u_ = np.dot(A_plus, Y_)

    return u_
    
def y_res(s,u_):
    return y_model(s,cn.u,cn.s_m) + y_model(s,u_[:cn.M_0],cn.s_m_0) + y_model(s,u_[cn.M_0:],cn.s_m_g)


def show_res():
    u_ = create_slar()
    fig2 = pylab.figure()
    axes = Axes3D(fig2)

    xx= np.arange (0, 1, 0.1)
    tt = np.arange (0, 1, 0.1)
    yy = np.zeros((xx.shape[0], tt.shape[0]))
    for i in range(xx.shape[0]):
        for j in range(tt.shape[0]):
            yy[i][j] = y_res([xx[i],tt[j]], u_)

    axes.plot_surface(xx,tt,yy, label = "y(s)")
    axes.set_xlabel("x")                              # подпись у горизонтальной оси х
    axes.set_ylabel("t")
    axes.set_zlabel("y")

    pylab.show()
    fig2.savefig('1_2.png')


if __name__ == '__main__':
    cn.test_observations()
    print(cn.s_0)
    show_constant()
    show_test_y()
    u_ = create_slar()
    show_res()
