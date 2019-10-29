import math
import numpy as np
import pylab
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# размерность пространства
N = 1

# время [0,T]
T = 1


# границы области пространства [ось][мин/макс значение]
scope = np.zeros((N,2))
for i in range(N):
    scope[0] = [0,1]

# исследуемая функция где s = [x, t], x = array[N эл]
def y(s: np.ndarray)-> float:
    res = 0
    #for i in range(s.shape[0]):
    for i in range(2):
        res+= (s[i] -0.5)**2
    return res

def sgn(x):
    if x >0:
        return 1
    elif x==0:
        return 0
    else:
        return -1

# функция Грина, пока просто пример
def G(s):
    if s[1] <= 0:
        return 0
    else:
        return 1/(4*math.pi*math.e**2*s[1])**0.5 *math.exp((-1*s[0]**2)/(4*math.e*s[1]))



#Надо будет как-то улучшить наблюдения
# начальные наблюдения
R_0 = 100
# массив с координатами наблюдений (столбец N+1 - время =0 )
s_0 = np.zeros((R_0, N+1))
Y_0 = np.zeros(R_0)
for i in range(R_0):
    s_0[i] = [i/R_0,0]
# заполняем массив Y_0
for i in range(R_0):
    Y_0[i] = y(s_0[i])


#граничные наблюдения
R_g = 100
# массив с координатами и временем наблюдений
s_g = np.zeros((R_g, N+1))
Y_g = np.zeros(R_g)
for i in range(R_g):
    s_g[i] = [i%2, i/100]

# заполняем массив Y_g
for i in range(R_g):
    Y_0[i] = y(s_g[i])


# моделирующая функция u
M = 100 
# массив с координатами и временем наблюдений
s_m = np.zeros((M, N+1))
for i in range(10):
    for j in range(10):
        s_m[i*10+j] = [i/10,j/10]
u = np.zeros(M)

def U(s):
    return 2*(s[1]-0.5)- 2*math.e**2

# заполняем массив u
for i in range(M):
    u[i] = U(s_m[i])


# можелирущая функция u_0
M_0 = 100
s_m_0 = np.zeros((M_0, N+1))
for i in range(10):
    for j in range(10):
        s_m_0[i*10+j] = [i/10+1/10,-j/10]


# можелирущая функция u_g
M_g = 100
s_m_g = np.zeros((M_g, N+1))
for i in range(10):
    for j in range(10):
        s_m_g[i*10+j] = [(1.1)*(-1)**(i%2)+(i)/10,j/10+1/10]



fig, axes = plt.subplots(figsize=(12,4))  

axes.scatter(s_0[:,0],s_0[:,1],s =0.1) # начальные наблюдения
axes.scatter(s_g[:,0],s_g[:,1], s=0.1) # граничные наблюдения
axes.scatter(s_m[:,0],s_m[:,1], s=5, marker="x") # возбуждения
axes.scatter(s_m_0[:,0],s_m_0[:,1], s=5, marker="o") # начальные возбуждения
axes.scatter(s_m_g[:,0],s_m_g[:,1], s=5) # граничные возбуждения
axes.set_xlabel("x")                              
axes.set_ylabel("t")
plt.legend(("s_0", "s_g", "s_m", "s_m_0", "s_m_g"))

#plt.show()
fig.savefig('1_0.png')


def makeData ():
    x = np.arange (0, 1, 0.1)
    y = np.arange (0, 1, 0.1)
    xgrid, ygrid = np.meshgrid(x, y)

    zgrid = (xgrid - 0.5)**2 + (ygrid - 0.5)**2
    return xgrid, ygrid, zgrid


fig1 = pylab.figure()
axes = Axes3D(fig1)

xx,tt,yy  = makeData()

axes.plot_surface(xx,tt,yy, label = "y(s)")
axes.set_xlabel("x")                              # подпись у горизонтальной оси х
axes.set_ylabel("t")
axes.set_zlabel("y")

#pylab.show()
fig1.savefig('1_1.png')


# вектор из u_0, u_g
u_ = np.zeros(M_0+M_g)

Y_0_1 = np.array([Y_0[i]- U(s_0[i]) for i in range(R_0)])
Y_g_1 = np.array([Y_g[i]- U(s_g[i]) for i in range(R_g)])
Y_ = np.append(Y_0_1,Y_g_1)

A_11 = np.zeros((R_0, M_0))
for i in range(R_0):
    for j in range(M_0):
        A_11[i][j] = G(s_0[i] - s_m_0[j])
 

A_12 = np.zeros((R_0, M_g))
for i in range(R_0):
    for j in range(M_g):
        A_12[i][j] = G(s_0[i] - s_m_g[j])
        


A_21 = np.zeros((R_g, M_0))
for i in range(R_g):
    for j in range(M_0):
        A_21[i][j] = G(s_g[i] - s_m_0[j])
        #print(s_g[i] - s_m_0[j], '  ', G(s_g[i] - s_m_0[j]))



A_22 = np.zeros((R_g, M_g))
for i in range(R_g):
    for j in range(M_g):
        A_22[i][j] = G(s_g[i] - s_m_g[j])


A_1 = np.append(A_11, A_12, axis= 1)
A_2 = np.append(A_21, A_22, axis= 1)
print(A_22)
A = np.append(A_1, A_2, axis= 0)

print((np.dot(A,np.transpose(A))))
A_plus = np.dot(np.transpose(A), np.linalg.inv(np.dot(A,np.transpose(A))))


v = np.ones(M_0+M_g)

u_ = np.dot(A_plus, (Y_ - np.dot(A,v))) +v

u_0 = u_[:M_0]

u_g = u_[M_0:]


# можелирущая функция s = [x,t] u_k, s_k  это соответсвенно u, s_m; u_0, s_m_0; u_g, s_m_g
def y_model(s, u_k, s_k):
    res=0
    for i in range(u_k.shape[0]):
        res+= G(s - s_k[i])*u_k[i]
    return res

print(u_0)

def Y_res(s):
    print(y_model(s,u,s_m),y_model(s,u_0,s_m_0) , y_model(s,u_g,s_m_g),sep ='  ')
    return y_model(s,u,s_m) + y_model(s,u_0,s_m_0) + y_model(s,u_g,s_m_g)


print(Y_res([0.5,0.5]))


fig2 = pylab.figure()
axes = Axes3D(fig2)

xx= np.arange (0, 1, 0.1)
tt = np.arange (0, 1, 0.1)
yy = np.zeros((xx.shape[0], tt.shape[0]))
for i in range(xx.shape[0]):
    for j in range(tt.shape[0]):
        yy[i][j] = Y_res([xx[i],tt[j]])

axes.plot_surface(xx,tt,yy, label = "y(s)")
axes.set_xlabel("x")                              # подпись у горизонтальной оси х
axes.set_ylabel("t")
axes.set_zlabel("y")

pylab.show(fig2)
fig2.savefig('1_2.png')