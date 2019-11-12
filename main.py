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
    t = np.arange (0, 1, 0.1)
    xx, tt = np.meshgrid(x, t)
    y = np.zeros((xx.shape[0], xx.shape[1]))
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            y[i][j] = fn.y([xx[i][j],tt[i][j]])

    fig1 = pylab.figure()
    axes = Axes3D(fig1)

    axes.plot_surface(xx,tt,y)
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

    Y_ = np.append(Y_0_new, Y_g_new, axis =0)

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

    P_1 = np.dot(A,np.transpose(A))

    # A_plus = np.dot(np.transpose(A), np.linalg.inv(np.dot(A,np.transpose(A))))

    A_plus = np.dot( np.transpose(A) , np.linalg.inv(P_1) )

    u_ = np.dot(A_plus, Y_)


    eps = np.dot(Y_, Y_) - np.dot( Y_,  np.dot( P_1, np.dot( np.linalg.inv(P_1), Y_ ) ) )
    return u_, eps
    
def y_res(s,u_):
    return y_model(s,cn.u,cn.s_m) + y_model(s,u_[:cn.M_0],cn.s_m_0) + y_model(s,u_[cn.M_0:],cn.s_m_g)


def show_res():
    u_,eps = create_slar()
    fig2 = pylab.figure()
    axes = Axes3D(fig2)

    xx = np.zeros(100)
    tt = np.zeros(100)
    for i in range(100):
        xx[i]= i/100
        

    

    x= np.arange (0, 1, 0.1)
    t = np.arange (0, 1, 0.1)
    xx, tt = np.meshgrid(x, t)
    yy = np.zeros((xx.shape[0], xx.shape[1]))
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            yy[i][j] = y_res([xx[i][j],tt[j][j]], u_)
    
            print(yy[i,j], '  ', fn.y([xx[i][j],tt[j][j]]), "ошибка: ", abs(yy[i,j]- fn.y([xx[i,j], tt[i,j]])))

    ## сравнение нулевых наблюдений

    print("сравнение нулевых наблюдений")
    for i in range(cn.R_0):
        print(y_res(cn.s_0[i], u_), '  ', fn.y(cn.s_0[i]), "ошибка: ", abs(y_res(cn.s_0[i], u_) - fn.y(cn.s_0[i])))

    ## сравнение краевых наблюдений


    print("сравнение краевых наблюдений")
    for i in range(cn.R_g):
        print(y_res(cn.s_g[i], u_), '  ', fn.y(cn.s_g[i]), "ошибка: ", abs(y_res(cn.s_g[i], u_) - fn.y(cn.s_g[i])))


    ## выводим ошибку

    print('Ошибка: {0:}'.format(eps)) 


    axes.plot_surface(xx,tt,yy, label = "y(s)")
    axes.set_xlabel("x")                              # подпись у горизонтальной оси х
    axes.set_ylabel("t")
    axes.set_zlabel("y")

    pylab.show()
    fig2.savefig('1_2.png')


if __name__ == '__main__':
    cn.test_observations_new()
    #print(cn.s_0)
    show_constant()
    show_test_y()
    u_ = create_slar()
    show_res()
