import matplotlib.pyplot as plt
import numpy as np
from qutip import *
from numpy.linalg import matrix_rank
from toqito.state_props import negativity
from scipy import integrate as integrate



# def state(r):
#     s=ket2dm(rand_ket(4*r))
#     s.dims=[[4,r],[4,r]]
#     ss=s.ptrace(0)
#     ss.dims=[[2,2],[2,2]]
#     return ss

def state(r):
    if r==1:
        rho=Qobj(np.matrix([[0.861553+0.0*1j,-0.189292+0.00294553*1j,0.135905-0.0936245*1j,0.173844+0.161189*1j],
                            [-0.189292-0.00294553*1j,0.0415995+0.0*1j,-0.0301798+0.0201056*1j,-0.0376444-0.0360094*1j],
                            [0.135905+0.0936245*1j,-0.0301798-0.0201056*1j,0.0316123+0.0*1j,0.00990655+0.0443182*1j],
                            [0.173844-0.161189*1j,-0.0376444+0.0360094*1j,0.00990655-0.0443182*1j,0.0652356+0.0*1j]]),dims=[[2,2],[2,2]])
    elif r==2:
        rho=Qobj(np.matrix([[0.076281+0.0*1j,-0.068415-0.146578*1j,0.0103318+0.0120563*1j,-0.0774808+0.00323882*1j],
                            [-0.068415+0.146578*1j,0.609089+0.0*1j,0.0562163-0.0605154*1j,-0.15758-0.1321*1j],
                            [0.0103318-0.0120563*1j,0.0562163+0.0605154*1j,0.0510243+0.0*1j,-0.0887119-0.038489*1j],
                            [-0.0774808-0.00323882*1j,-0.15758+0.1321*1j,-0.0887119+0.038489*1j,0.263606+0.0*1j]]),dims=[[2,2],[2,2]])
    elif r==3:
        rho=Qobj(np.matrix([[0.452554+0.0*1j,-0.137758+0.0173835*1j,0.0909171-0.110103*1j,0.0553365-0.16667*1j],
                            [-0.137758-0.0173835*1j,0.0894731+0.0*1j,-0.0505617+0.0331408*1j,-0.00604473+0.13467*1j],
                            [0.0909171+0.110103*1j,-0.0505617-0.0331408*1j,0.077898+0.0*1j,0.0434916+0.00517076*1j],
                            [0.0553365+0.16667*1j,-0.00604473-0.13467*1j,0.0434916-0.00517076*1j,0.380075+0.0*1j]]),dims=[[2,2],[2,2]])
    elif r==4:
        rho=Qobj(np.matrix([[0.139614+0.0*1j,-0.059128+0.0722803*1j,-0.0216255-0.010379*1j,-0.0380448-0.121063*1j],
                            [-0.059128-0.0722803*1j,0.142448+0.0*1j,-0.068477+0.155433*1j,-0.0532428+0.122709*1j],
                            [-0.0216255+0.010379*1j,-0.068477-0.155433*1j,0.388016+0.0*1j,0.096868+0.0504258*1j],
                            [-0.0380448+0.121063*1j,-0.0532428-0.122709*1j,0.096868-0.0504258*1j,0.329921+0.0*1j]]),dims=[[2,2],[2,2]])
    return rho

def wootters(r,x1,p1,x2,p2):
    kernel1=(1.0/2.0)*(qeye(2)+((-1.)**x1) * sigmaz()+((-1.)**p1) * sigmax()+((-1.)**(x1+p1)) * sigmay())
    kernel2=(1.0/2.0)*(qeye(2)+((-1.)**x2) * sigmaz()+((-1.)**p2) * sigmax()+((-1.)**(x2+p2)) * sigmay())
    w=(1./4.)*(state(r)*tensor(kernel1,kernel2)).tr()
    return np.real(w) 

def tilma(r,t1,f1,t2,f2):
    kernel1=1./2. * (qeye(2)-np.sqrt(3.)*(np.cos(2.*t1)*sigmaz()+np.sin(2.*t1)*(np.cos(2.*f1)*sigmax()+np.sin(2*f1)*sigmay())))
    kernel2=1./2. * (qeye(2)-np.sqrt(3.)*(np.cos(2.*t2)*sigmaz()+np.sin(2.*t2)*(np.cos(2.*f2)*sigmax()+np.sin(2*f2)*sigmay())))
    w=(state(r)*tensor(kernel1,kernel2)).tr()
    return np.real(w)   



# options={'limit':1}
# print(integrate.nquad(lambda t1,t2,f1,f2: tilma(1,t1,f1,t2,f2),[[0,np.pi/2],[0,np.pi/2],[0,2*np.pi],[0,2*np.pi]],opts=[options,options,options,options]))

r=4
summ=[]
for x1 in [0,1]:
    for p1 in [0,1]:
        for x2 in [0,1]:
            for p2 in [0,1]:
                summ+=[wootters(r,x1,p1,x2,p2)]
print(sum(summ))

# for x in np.linspace(0,1,100):
#     print(matrix_rank(rand_dm(N=4,density=x)))


# for x in rank:
#     c+=[concurrence(state(x))]
#     n+=[negativity(state(x))]
# plt.scatter(n,c)
# plt.show()