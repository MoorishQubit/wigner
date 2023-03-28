import matplotlib.pyplot as plt
import numpy as np
from qutip import *
from numpy.linalg import matrix_rank
from toqito.state_props import negativity
from scipy import integrate as integrate
import itertools

plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['mathtext.fontset'] = 'stix'

marker = itertools.cycle(('o', '+', 'x','*')) 


def state(r):
    s=ket2dm(rand_ket_haar(4*r))
    s.dims=[[4,r],[4,r]]
    ss=s.ptrace(0)
    ss.dims=[[2,2],[2,2]]
    return ss

# def state(r):
#     return rand_dm_ginibre(4, rank=r,dims=[[2,2], [2,2]])

def wootters(r,x1,p1,x2,p2):
    kernel1=(1.0/2.0)*(qeye(2)+((-1.)**x1) * sigmaz()+((-1.)**p1) * sigmax()+((-1.)**(x1+p1)) * sigmay())
    kernel2=(1.0/2.0)*(qeye(2)+((-1.)**x2) * sigmaz()+((-1.)**p2) * sigmax()+((-1.)**(x2+p2)) * sigmay())
    w=(1./4.)*(state(r)*tensor(kernel1,kernel2)).tr()
    return np.real(w) 

def tilma(r,t1,f1,t2,f2):
    kernel1=1./2. * (qeye(2)-np.sqrt(3.)*(np.cos(2.*t1)*sigmaz()+np.sin(2.*t1)*(np.cos(2.*f1)*sigmax()+np.sin(2*f1)*sigmay())))
    kernel2=1./2. * (qeye(2)-np.sqrt(3.)*(np.cos(2.*t2)*sigmaz()+np.sin(2.*t2)*(np.cos(2.*f2)*sigmax()+np.sin(2*f2)*sigmay())))
    w=(state(r)*tensor(kernel1,kernel2)).tr()
    return np.abs(np.real(w))-np.real(w)   

def negativity_wootters(r):
    summ=[]
    for x1 in [0,1]:
        for p1 in [0,1]:
            for x2 in [0,1]:
                for p2 in [0,1]:
                    summ+=[wootters(r,x1,p1,x2,p2)]
    return sum(summ) 

# def negativity_tilma(r):
#     #options={'limit':5}
#     return integrate.nquad(lambda t,f: tilma(r,t,f,t,f)*np.sin(2*t)/np.pi,[[0,np.pi/2],[0,2*np.pi]])#,opts=[options,options])[0]

print(state(1))

# for x in np.linspace(0,1,100):
#     print(matrix_rank(rand_dm(N=4,density=x)))

# c=[]
# n=[]
# for x in [1,2,3,4]:
#     c=[concurrence(state(x))]
#     n=[negativity(state(x))]
#     plt.scatter(n,c,label=r'rank $%s$'%x,marker=next(marker))
# plt.legend(fontsize=20,loc='lower right')
# plt.xlabel(r'Negativity Wootters',fontsize=20)
# plt.ylabel(r'Entanglement Negativity',fontsize=20)
# plt.tick_params(axis='both', which='major', labelsize=15)

# plt.tight_layout()
# plt.show()
