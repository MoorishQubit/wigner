import matplotlib.pyplot as plt
import numpy as np
from qutip import Qobj, concurrence, bell_state, ket2dm, qeye, tensor
plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['mathtext.fontset'] = 'stix'

def werner_state(l):
    state = l * tensor(qeye(2),qeye(2))/4 + (1-l)*ket2dm(bell_state('11'))
    state.dims=[[2,2],[2,2]]
    return concurrence(state) #state.purity()

def concurrence_werner(l):
    f1=abs((1-l)/2) - l/4
    f2=- l/4 - (1-l)/2
    return 2*max(0,f1,f2)

def negativity_wigner_werner(l):
    f1=3*(1+(1-l))
    f2=1-3*(1-l)
    return 1/4 * (abs(f1)+abs(f2)) - 1

l = np.linspace(0,4/3,100)
plt.plot(l, list(map(lambda l:werner_state(l),l)),'-*',label=r'$C_W$')
#plt.plot(l, list(map(lambda l:werner_state(l),l)),'-*',label=r'$P_W$')
plt.plot(l, list(map(lambda l:negativity_wigner_werner(l),l)),'--',label=r'$N_W$')
plt.legend(fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout()
plt.savefig("cnc_negW.pdf")
plt.show()

#print(ket2dm(bell_state('11')))