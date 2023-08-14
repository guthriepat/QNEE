import numpy as np 
from qiskit import quantum_info

def initial_entangle_EP(ini_counts, num_qr):
    """
    input
    ini_counts: class
        backend.run.result().get_counts() object 
    num_qr: int 
        initial qubit number 
    return 
    entangled_entropy: float 
    """
    initial_state_vec = []
    n_redueced_num = int(num_qr/2)
    #print(n_redueced_num)
    for i in np.arange(num_qr):
        for j in np.arange(num_qr):
            string = str(i)+str(j)
            initial_state_vec.append(ini_counts[string])
    initial_state_vec =np.sqrt( np.array(initial_state_vec)/np.sum(initial_state_vec))
    reduced_density_mat = quantum_info.partial_trace(initial_state_vec, (num_qr - 1) - np.arange(n_redueced_num))
    #print(reduced_density_mat)
    entangled_entropy = quantum_info.entropy(reduced_density_mat,base=np.e)
    return entangled_entropy

def data_processing(counts, n_qubit):
    """
    input
    ini_counts: class
        backend.run.result().get_counts() object 
    n_quibt: # of observed qubit! 
    
    returns
        data, data_table, entropy 
        data_table is a list of keys (tokens)
    """
    keylist = list(counts.keys())
    data_table = []
    for i in keylist:
        data_table.append(int(i[n_qubit:],2))
    #print(data_table)
    data = np.array([])
    entropy = 0
    total_n = np.sum(list(counts.values()))
    for i in keylist:
        temp_array = int(i[n_qubit:],2) * np.ones(counts[i],dtype='int')
        prop = counts[i]/ total_n
        entropy += -prop * np.log(prop)
        data = np.concatenate([data,temp_array] )
    return data, data_table, entropy 