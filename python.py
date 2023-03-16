import matplotlib.pyplot as plt
import numpy as np
from qutip import *
from numpy.linalg import matrix_rank
rank=np.linspace(0,1,100)
c=[]
n=[]
for x in rank:
    #print(rand_dm(4,density=x,dims=[[2,2],[2,2]]))
    #print(matrix_rank(rand_dm(4,density=x,dims=[2*[2],2*[2]])))
    c+=[concurrence(rand_dm(4,density=x,dims=[[2,2],[2,2]]))]
    n+=[negativity(rand_dm(4,density=x,dims=[[2,2],[2,2]]),2)]
plt.scatter(n,c)
plt.show()