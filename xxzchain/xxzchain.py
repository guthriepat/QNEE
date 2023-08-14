import numpy as np
import qutip as qt
import time


def groundXXZ(num_spins,delta,magfield):
    """
    Not recommend more than 12 quibts 
    Hamiltonian in PRL 90 227902
    \sigma_l^x\sigma_{l+1}^x + \sigma_l^y\sigma_{l+1}^y + \delta  \sigma_l^z\sigma_{l+1}^z 
    - \lambda \sigma_l^z 
    |010> means |up down up>
    input 
        num_spins: total number of spins
        delta: real number
        magfield: real number
    output 
        Ground state! qutip object! 
    """
    
    t0 = time.time()
    Ham = 0.
    for i in np.arange(num_spins ):
        #print(i)
        Ham += qt.tensor([ qt.sigmax() if j == i or j==(i+1)%num_spins  else qt.qeye(2) 
                          for j in np.arange(num_spins)]) #qt.sigmax(),qt.sigmax() )
        Ham += qt.tensor([ qt.sigmay() if j == i or j==(i+1)%num_spins   else qt.qeye(2) 
                          for j in np.arange(num_spins)]) 
        Ham += delta * qt.tensor([ qt.sigmaz() if j == i or j==(i+1)%num_spins   else qt.qeye(2) 
                          for j in np.arange(num_spins)]) 
        Ham += - magfield * qt.tensor([ qt.sigmaz() if j == i else qt.qeye(2) 
                          for j in np.arange(num_spins)]) 
    t1 =time.time()

    #print(t1-t0)
    #print(Ham)
    thestate = Ham.groundstate()[-1]
    #print(Ham.eigenstates())
    return thestate

def entangleandvislist(thestate, num_spins):
    """
    input
        thestate = ground state 
        num_spins 
    """
    visible_list = []
    entangle_list = []
    for m in np.arange(int(num_spins/2)):
        visible_sites = m+1
        tempt = time.time()
        reducedmat = qt.ptrace(thestate,[elem for elem in np.arange(visible_sites)])
        temp_ent = qt.entropy_vn(reducedmat, base=np.e)
        entangle_list.append(temp_ent)
        visible_list.append(visible_sites)
    return entangle_list, visible_list

def xxzansatz(x,a,b):
    """
    use as 
    popt2, pcov2 = curve_fit(func2, visible_list, entangle_list)
    """
    return a / 3. * np.log(num_spins/np.pi*np.sin(x*np.pi/num_spins) ) + b