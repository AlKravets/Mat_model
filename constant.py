import math
import numpy as np
import function as fn
# размерность пространства
N = 1

# время [0,T]
T = 1


# размерность нового пространства
S = N+1


# границы области пространства [ось][мин/макс значение]
scope = np.zeros((S,2))
scope[-1] = [0,T]

# начальные наблюдения
R_0 = 100
# массив с координатами наблюдений (столбец N+1 - время =0 )
s_0 = np.zeros((R_0, S))
Y_0 = np.zeros(R_0)


#граничные наблюдения
R_g = 100
# массив с координатами и временем наблюдений
s_g = np.zeros((R_g, S))
Y_g = np.zeros(R_g)


# моделирующая функция u
M = 100 
# массив с координатами и временем наблюдений
s_m = np.zeros((M, N+1))

u = np.zeros(M)


# можелирущая функция u_0
M_0 = 100
s_m_0 = np.zeros((M_0, N+1))


# можелирущая функция u_g
M_g = 100
s_m_g = np.zeros((M_g, N+1))

def test_observations_new():
    global N,T,S,scope, R_0, s_0, Y_0, R_g, s_g, Y_g, M, s_m, u, M_0, s_m_0, M_g, s_m_g

    N=1
    
    T = 1
    
    S = N+1
    
    scope = np.zeros((S,2))
    scope[-1] = [0,T]
    for i in range(N):
        scope[i] = [0,1]
    
    R_0 = 5
    
    s_0 = np.zeros((R_0, S))
    Y_0 = np.zeros(R_0)
    
    for i in range(R_0):
        s_0[i] = [i/R_0,0]
    
    for i in range(R_0):
        Y_0[i] = fn.y(s_0[i])

    
    R_g = 5
    
    s_g = np.zeros((R_g, S))
    Y_g = np.zeros(R_g)
    

    for i in range(R_g):
        s_g[i] = [i%2, (i+1)/R_g]

    for i in range(R_g):
        Y_g[i] = fn.y(s_g[i])
    
    
    M = 10 
    
    s_m = np.zeros((M, S))
    
    for i in range(5):
        for j in range(2):
            s_m[i*2+j] = [i/5,j/4+0.1]
    u = np.zeros(M)

    for i in range(M):
        u[i] = fn.U(s_m[i])
    
    M_0 = 10
    s_m_0 = np.zeros((M_0, S))
    for i in range(5):
        for j in range(2):
            s_m_0[i*2+j] = [i/10+1/10,-j/10-0.1]


    M_g = 10
    s_m_g = np.zeros((M_g, S))
    for i in range(5):
        for j in range(2):
            s_m_g[i*2+j] = [(1.1)*(-1)**(i%2)+(i)/10,j/10+1/10]



def test_observations():
    global N,T,S,scope, R_0, s_0, Y_0, R_g, s_g, Y_g, M, s_m, u, M_0, s_m_0, M_g, s_m_g

    N=1
    
    T = 1
    
    S = N+1
    
    scope = np.zeros((S,2))
    scope[-1] = [0,T]
    for i in range(N):
        scope[i] = [0,1]
    
    R_0 = 100
    
    s_0 = np.zeros((R_0, S))
    Y_0 = np.zeros(R_0)
    
    for i in range(R_0):
        s_0[i] = [i/R_0,0]
    
    for i in range(R_0):
        Y_0[i] = fn.y(s_0[i])

    
    R_g = 100
    
    s_g = np.zeros((R_g, S))
    Y_g = np.zeros(R_g)
    

    for i in range(R_g):
        s_g[i] = [i%2, i/100]

    for i in range(R_g):
        Y_g[i] = fn.y(s_g[i])
    
    
    M = 100 
    
    s_m = np.zeros((M, S))
    
    for i in range(10):
        for j in range(10):
            s_m[i*10+j] = [i/10,j/10]
    u = np.zeros(M)

    for i in range(M):
        u[i] = fn.U(s_m[i])
    
    M_0 = 100
    s_m_0 = np.zeros((M_0, S))
    for i in range(10):
        for j in range(10):
            s_m_0[i*10+j] = [i/10+1/10,-j/10]


    M_g = 100
    s_m_g = np.zeros((M_g, S))
    for i in range(10):
        for j in range(10):
            s_m_g[i*10+j] = [(1.1)*(-1)**(i%2)+(i)/10,j/10+1/10]




if __name__ == '__main__':
    test_observations()
    print(s_0)
