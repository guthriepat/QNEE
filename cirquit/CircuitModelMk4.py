# In this mark4 version, I update the circuit asatz !! 
import numpy as np
# quantum circuit 
from qiskit import Aer
import copy
backend = Aer.get_backend("aer_simulator") # set simulator
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from cirquit.tools import data_processing, initial_entangle_EP
from qiskit.algorithms.optimizers import COBYLA, ADAM, SLSQP, SPSA, GradientDescent
from qiskit.quantum_info import random_statevector, random_density_matrix
from qiskit.circuit.library import EfficientSU2
from torch.utils.data import DataLoader
from neural.net import VonNeumannEP, train, validate
import torch 
#from qiskit import tools
from qiskit import quantum_info

import qutip as qt

import time

from qiskit import IBMQ
#### IBMQ(https://quantum-computing.ibm.com/) account might be necessary 
# please type your account 
#IBMQ.save_account('account')


class HybridCircuit:
    def __init__(self, opt):
        self.n_qubit = opt.n_qubit # number of qubits that we observe
        self.n_clbit = opt.n_clbit # number of classical bits that we observe
        self.ini_state = opt.ini_state # initial state
        self.reps = opt.reps # circuit depth
        self.n_shots = opt.n_shots # shots for one ensemble
        self.num_ini_qubit = opt.num_ini_qubit # number of initial qubits
        self.num_ini_clbit = opt.num_ini_clbit # number of inital classical bits
        self.n_token = opt.n_token
        
        self.n_hidden = opt.n_hidden # number of hidden variables of nn
        self.n_layer  = opt.n_layer  # number of hiddlen layers of nn
        self.batch_size = opt.batch_size  # batch size
        self.device = opt.device  # device
        self.lr = opt.lr  # learning rate
        self.wd = opt.wd  # weight decay
        self.record_freq = opt.record_freq # record frequency  
        self.seed = opt.seed # seed for nn
        self.datasize = opt.datasize  # data size
        self.model = opt.model # nn model. 
        self.inter_nn_iter = opt.inter_nn_iter # training for cost ftn.
        
        #self.ftn = optqc.f 
        
        self.optim = opt.optim 
        
    def build_qc(self, parameters):
        q = QuantumRegister(self.num_ini_qubit, name="q")
        c = ClassicalRegister(self.num_ini_clbit, name="c")
        qc = QuantumCircuit(q, c)
        qc.initialize(self.ini_state)
        shift = 0 
        #####################################
        for i in np.arange(self.n_qubit):
            k = i
            qc.ry(parameters[k],q[i])
        aux_par_num = k+1 
        for i in np.arange(int(self.n_qubit/2)):
            qc.cz(q[2*i+shift],q[(2*i+1+shift)%self.n_qubit])
        for i in np.arange(self.n_qubit):
            k =i + aux_par_num
            qc.ry(parameters[k],q[i])
        aux_par_num = k+1    
        shift+=1
        shift%=2
        for j in np.arange(self.reps-1):
        #####################################
            for i in np.arange(self.n_qubit):
                k =i + aux_par_num
                qc.ry(parameters[k],q[i])
            aux_par_num = k+1  
            for i in np.arange(int(self.n_qubit/2)):
                qc.cz(q[2*i+shift],q[(2*i+1+shift)%self.n_qubit])
            #####################################
            for i in np.arange(self.n_qubit):
                k =i + aux_par_num
                qc.ry(parameters[k],q[i])
            aux_par_num = k+1    
            shift+=1
            shift%=2
        qc.save_statevector() 
        qc.measure(q[:self.n_qubit], c[:self.n_qubit])
        return qc 
    def costfunction(self, parameters):
        #print('start')
        ######## train data ######################### 
        qc = self.build_qc(parameters)
        result = backend.run(qc, shots = self.n_shots).result()
        _counts = result.get_counts()
        _data, _data_table, entropy = data_processing(_counts, self.n_qubit)
        # data loader 를 한 개만 쓰고 dataloader.data = (?) 방식으로 안에 data를 바꿔준다!
        _train_dataloader = DataLoader(_data, 
                                       batch_size = self.batch_size, 
                                       shuffle = True)
        ########## test data ##################################
        result = backend.run(qc, shots = self.n_shots).result()
        _counts = result.get_counts()
        test_data, test_data_table, entropy = data_processing(_counts, self.n_qubit) 
        test_dataloader  = DataLoader(test_data, 
                                      batch_size = self.n_shots, 
                                      shuffle=False)

        ########### training ######################## 
        test_cost, _ = validate(self, self.model, 
                                test_data_table,
                                test_dataloader)
        
        loss_list = []
        self.save_NEEP_output = test_cost.item()
        record_value = test_cost
        result = test_cost
        for i in np.arange(1, self.inter_nn_iter + 1):
            temp_loss, temp_normal = train(self, # <- self.opt works if i add self.opt = opt 
                                           self.model,_data_table,  
                                          self.optim , _train_dataloader)
            test_cost, _ = validate(self, self.model, 
                                test_data_table,
                                test_dataloader)
            if test_cost[0] < self.save_NEEP_output:
                #print('save output: ', test_cost)
                self.save_NEEP_output = copy.deepcopy(test_cost.item())
                self.saveAdamState = copy.deepcopy(self.optim.state_dict())
                self.saveNNState   = copy.deepcopy(self.model.state_dict())
            
            if i% self.record_freq == 0:
                if record_value > test_cost:
                    record_value = test_cost
                else:
                    self.model.load_state_dict(self.saveNNState)
                    self.optim.load_state_dict(self.saveAdamState) # something wrong... ? or work?
                    break 
                    
        return self.save_NEEP_output # temp_loss 
    
    def evaluationQC(self, parameters):
        # print('eval')
        ######## eval data ######################### 
        qc = self.build_qc(parameters)
        result = backend.run(qc, shots = self.n_shots).result()
        _counts = result.get_counts()
        _data, _data_table, entropy = data_processing(_counts, self.n_qubit)
        _validate_dataloader = DataLoader(_data, 
                                         batch_size = self.n_shots, 
                                         shuffle = False)
        
        cost, _ = validate(self, self.model, _data_table,_validate_dataloader)
        return cost.item() #.to('cpu').numpy()
    def exactRenyicostfunction(self,parameters):
        _alpha = self.alpha # added term for Renyi cost ftn 
        qc = self.build_qc(parameters)
        result = backend.run(qc, shots = 1).result()
        _counts = result.get_counts()
        _data, _data_table, entropy = data_processing(_counts, self.n_qubit)
        _fin_statevec  = result.get_statevector(qc)
        _fin_reduced_state = quantum_info.partial_trace(_fin_statevec,
                                                       list(self.n_qubit + np.arange(self.n_qubit)))
        test_sum = 0
        cost_ftn = 0
        for i in np.arange(2**_fin_reduced_state.num_qubits):
            for j in np.arange(2**_fin_reduced_state.num_qubits):
                if i==j:
                    pii = _fin_reduced_state.data[i,i]
                    pii = pii.real
                    cost_ftn += pii ** _alpha
        cost_ftn /= _alpha *(1-_alpha)
        return cost_ftn
    def exactcostfunction(self,parameters):
        qc = self.build_qc(parameters)
        result = backend.run(qc, shots = 1).result()
        _counts = result.get_counts()
        _data, _data_table, entropy = data_processing(_counts, self.n_qubit)
        _fin_statevec  = result.get_statevector(qc)
        _fin_reduced_state = quantum_info.partial_trace(_fin_statevec,
                                                        list(np.arange(self.n_qubit,self.num_ini_qubit) )# remove quits n_qubits+1 ~ ini_qubits
                                                       )
        test_sum = 0
        cost_ftn = 0
        for i in np.arange(2**_fin_reduced_state.num_qubits):
            for j in np.arange(2**_fin_reduced_state.num_qubits):
                if i == j:
                    pii = _fin_reduced_state.data[i,i]
                    pii = pii.real
                    test_sum += pii
                    cost_ftn += - np.log( pii ) * pii
        cost_ftn += np.log(test_sum)
        return cost_ftn
    def estimateeigenval(self,parameters):
        qc = self.build_qc(parameters)
        result = backend.run(qc, shots = self.n_shots).result()
        _counts = result.get_counts()
        _data, _data_table, entropy = data_processing(_counts, self.n_qubit)

        return np.sort(np.array(list(_counts.values()))/np.sum(list(_counts.values())))[::-1]
    
class VQSE:
    def __init__(self, opt):
        self.n_qubit = opt.n_qubit # number of qubits that we observe
        self.n_clbit = opt.n_clbit # number of classical bits that we observe
        self.ini_state = opt.ini_state # initial state
        self.reps = opt.reps # circuit depth
        self.n_shots = opt.n_shots # shots for one ensemble
        self.num_ini_qubit = opt.num_ini_qubit # number of initial qubits
        self.num_ini_clbit = opt.num_ini_clbit # number of inital classical bits
        self.n_token = opt.n_token
        
        # added 6th June 
        # for VQSE cost ftn
        self.r1 = opt.r1
        self.VQSEdelta = opt.VQSEdelta
        self.GLratio = opt.GLratio      
        self.optim = opt.optim 
        try:
            self.m_indexlist = opt.m_indexlist
        except:
            print('no m_lindexlist')
        try:
            self.m_keylist = opt.m_keylist
        except:
            print('no m_keylist')
        
    def build_qc(self, parameters):
        q = QuantumRegister(self.num_ini_qubit, name="q")
        c = ClassicalRegister(self.num_ini_clbit, name="c")
        qc = QuantumCircuit(q, c)
        qc.initialize(self.ini_state)
        shift = 0 
        #####################################
        for i in np.arange(self.n_qubit):
            k = i
            qc.ry(parameters[k],q[i])
        aux_par_num = k+1 
        for i in np.arange(int(self.n_qubit/2)):
            qc.cz(q[2*i+shift],q[(2*i+1+shift)%self.n_qubit])
        for i in np.arange(self.n_qubit):
            k =i + aux_par_num
            qc.ry(parameters[k],q[i])
        aux_par_num = k+1    
        shift+=1
        shift%=2
        for j in np.arange(self.reps-1):
        #####################################
            for i in np.arange(self.n_qubit):
                k =i + aux_par_num
                qc.ry(parameters[k],q[i])
            aux_par_num = k+1  
            for i in np.arange(int(self.n_qubit/2)):
                qc.cz(q[2*i+shift],q[(2*i+1+shift)%self.n_qubit])
            #####################################
            for i in np.arange(self.n_qubit):
                k =i + aux_par_num
                qc.ry(parameters[k],q[i])
            aux_par_num = k+1    
            shift+=1
            shift%=2
        qc.save_statevector() 
        qc.measure(q[:self.n_qubit], c[:self.n_qubit])
        return qc 
    def exactHamcostfunction(self,parameters):
        qc = self.build_qc(parameters)
        result = backend.run(qc, shots = 1).result()
        _counts = result.get_counts()
        _data, _data_table, entropy = data_processing(_counts, self.n_qubit)
        _fin_statevec  = result.get_statevector(qc)
        _fin_reduced_state = quantum_info.partial_trace(_fin_statevec,
                                                        list(np.arange(self.n_qubit,self.num_ini_qubit) )# remove quits n_qubits+1 ~ ini_qubits
                                                       )

        
        Hl, _r_list = self.gen_Hl(_fin_reduced_state.num_qubits, 
                                  _fin_reduced_state)
        #print(Hl, _r_list)
        
        
        m = _fin_reduced_state.num_qubits + 1 ## m = n+1 ! look the paper VQSE 
        q_list = self.gen_qlist(m,_r_list) # large population * large q! (small E )
        #print('q_list',q_list)
        

        Hg = 1
        for idx, k in enumerate(self.m_indexlist):
            diag_comp = _fin_reduced_state.data[k,k].real
            cof = q_list[idx]
            Hg -=  cof * diag_comp


        cost_ftn = Hg * self.GLratio + Hl * (1 - self.GLratio)        
        return cost_ftn 

    def exactRenyicostfunction(self,parameters):
        _alpha = self.alpha # added term for Renyi cost ftn 
        qc = self.build_qc(parameters)
        result = backend.run(qc, shots = 1).result()
        _counts = result.get_counts()
        _data, _data_table, entropy = data_processing(_counts, self.n_qubit)
        _fin_statevec  = result.get_statevector(qc)
        _fin_reduced_state = quantum_info.partial_trace(_fin_statevec,
                                                       list(self.n_qubit + np.arange(self.n_qubit)))
        test_sum = 0
        cost_ftn = 0
        for i in np.arange(2**_fin_reduced_state.num_qubits):
            for j in np.arange(2**_fin_reduced_state.num_qubits):
                if i==j:
                    pii = _fin_reduced_state.data[i,i]
                    pii = pii.real
                    cost_ftn += pii ** _alpha
        cost_ftn /= _alpha *(1-_alpha)
        return cost_ftn
    def exactcostfunction(self,parameters):
        qc = self.build_qc(parameters)
        result = backend.run(qc, shots = 1).result()
        _counts = result.get_counts()
        _data, _data_table, entropy = data_processing(_counts, self.n_qubit)
        _fin_statevec  = result.get_statevector(qc)
        _fin_reduced_state = quantum_info.partial_trace(_fin_statevec,
                                                        list(np.arange(self.n_qubit,self.num_ini_qubit) )# remove quits n_qubits+1 ~ ini_qubits
                                                       )
        test_sum = 0
        cost_ftn = 0
        for i in np.arange(2**_fin_reduced_state.num_qubits):
            for j in np.arange(2**_fin_reduced_state.num_qubits):
                if i == j:
                    pii = _fin_reduced_state.data[i,i]
                    pii = pii.real
                    test_sum += pii
                    cost_ftn += - np.log( pii ) * pii
        cost_ftn += np.log(test_sum)
        return cost_ftn
    def gen_qlist(self,m, r_list):
        """
        create qlist for Hg from r_list of Hl 
        """
        LocalHam =  qt.tensor([  qt.qeye(2) for j in np.arange(self.n_qubit )])
        for i in np.arange(self.n_qubit ):
            LocalHam += -  r_list[i] * qt.tensor([ (1/2) * qt.sigmaz() if j == i else qt.qeye(2) 
                                                for j in np.arange(self.n_qubit )])
        result = 1 - LocalHam.eigenenergies()[:m]
        return result 
    def gen_Hl(self,num_view_qubits, partial_state):
        """
        returns
        result = Hl value
        r_list = r value list, for cal of Hl and Hg
        """
        
        Hl = 1.
        traceoverlist = np.arange(num_view_qubits) 
        #print(traceoverlist)
        _updownlist = []
        for k in np.arange(num_view_qubits):
            l = list(traceoverlist)
            l.remove(k)
            #print(l)
            _temp_state = quantum_info.partial_trace(partial_state,
                                                     l# remove quits n_qubits+1 ~ ini_qubits
                                                     )
            assert _temp_state.num_qubits == 1 
            _updownlist.append( (_temp_state.data[0,0].real, _temp_state.data[1,1].real) )
        #print(_updownlist)
        r_list = [] # build r list for coef of Sz 
        for k in  np.arange(self.n_qubit):
            val = self.r1 + self.VQSEdelta * (k)
            r_list.append(val)
        
        for idx, rho in enumerate(_updownlist): # Sz evaluation
            Hl+= - r_list[idx] * ( rho[0] - rho[1] )/2 # i added half factor 
        #print(idx, 'idx!!', rho[0], 'population!!' )
        return Hl, r_list 
    def gen_idxlist(self, reduced_state):
        _, _r_list = self.gen_Hl(reduced_state.num_qubits, 
                                  reduced_state)
        
        m = reduced_state.num_qubits + 1 ## m = n+1 ! look the paper VQSE 
        
        evallist = []
        for i in np.arange(2**reduced_state.num_qubits):
            evallist.append(reduced_state.data[i,i].real)
        evallist = np.array(evallist)
        mevallist = evallist.copy()
        mevallist.sort() 
        mevallist = mevallist[::-1][:m] # m diaognal element from lagest one! 
        mindex_list = []
        for i in mevallist:
            mindex_list.append(np.where(evallist ==i )[0].squeeze() )
        return mindex_list, mevallist
    def finite_gen_Hl(self, counts):
        r_list = [] # build r list for coef of Sz 
        for k in  np.arange(self.n_qubit):
            val = self.r1 + self.VQSEdelta * (k)
            r_list.append(val)

        print(r_list)

        temp_keys = []
        energy_list = []
        for key in list(counts.keys()):
            key = key[-self.n_qubit:]
            temp_keys.append( key )
            energy = 0
            for idx, state in enumerate(key):
                #print(state)
                state = - int(state)
                state += 1/2.
                #print(state)
                energy += state * r_list[idx]
            energy_list.append(energy)
        #print(temp_keys)
        #print(energy_list)
        proplist = list(counts.values()) 
        proplist /= np.sum(proplist)
        #print(proplist)
        Hl = np.inner( proplist , energy_list)
        return Hl, r_list 
    def finite_gen_keylist(self, counts):
        test = np.array(list(counts.values()))
        test.sort()
        test = test[::-1]
        #print(test[:opt.n_qubit+1]) # sort? 
        mkeys = []
        for _, cont in enumerate(test[:self.n_qubit+1]):
            idx = np.where( np.array(list(counts.values())) == cont)[0][0]
            mkeys.append(list(counts.keys())[idx])

        return mkeys
    
    def finiteHamcostfunction(self,parameters):
        qc = self.build_qc(parameters)
        result = backend.run(qc, shots = self.n_shots).result()
        _counts = result.get_counts()
        
        Hl, _r_list = self.finite_gen_Hl(_counts)
       
        m = self.n_qubit + 1 ## m = n+1 ! look the paper VQSE 
        q_list = self.gen_qlist(m,_r_list) # large population * large q! (small E )
        print('q_list',q_list)
        

        Hg = 1
        for idx, k in enumerate(self.m_keylist):
            diag_comp =_counts[k]/self.n_shots
            cof = q_list[idx]
            print(diag_comp, cof)
            Hg -=  cof * diag_comp


        cost_ftn = Hg * self.GLratio + Hl * (1 - self.GLratio)        
        return cost_ftn     
    

#class VQSE:
#    def __init__(self, opt):
#        self.n_qubit = opt.n_qubit # number of qubits that we observe
#        self.n_clbit = opt.n_clbit # number of classical bits that we observe
#        self.ini_state = opt.ini_state # initial state
#        self.reps = opt.reps # circuit depth
#        self.n_shots = opt.n_shots # shots for one ensemble
#        self.num_ini_qubit = opt.num_ini_qubit # number of initial qubits
#        self.num_ini_clbit = opt.num_ini_clbit # number of inital classical bits
#        self.n_token = opt.n_token
#        
#        # added 6th June 
#        # for VQSE cost ftn
#        self.r1 = opt.r1
#        self.VQSEdelta = opt.VQSEdelta
#        self.GLratio = opt.GLratio      
#        self.optim = opt.optim 
#        try:
#            self.m_indexlist = opt.m_indexlist
#        except:
#            print('no m_lindexlist')
#        
#    def build_qc(self, parameters):
#        q = QuantumRegister(self.num_ini_qubit, name="q")
#        c = ClassicalRegister(self.num_ini_clbit, name="c")
#        qc = QuantumCircuit(q, c)
#        qc.initialize(self.ini_state)
#        shift = 0 
#        #####################################
#        for i in np.arange(self.n_qubit):
#            k = i
#            qc.ry(parameters[k],q[i])
#        aux_par_num = k+1 
#        for i in np.arange(int(self.n_qubit/2)):
#            qc.cz(q[2*i+shift],q[(2*i+1+shift)%self.n_qubit])
#        for i in np.arange(self.n_qubit):
#            k =i + aux_par_num
#            qc.ry(parameters[k],q[i])
#        aux_par_num = k+1    
#        shift+=1
#        shift%=2
#        for j in np.arange(self.reps-1):
#        #####################################
#            for i in np.arange(self.n_qubit):
#                k =i + aux_par_num
#                qc.ry(parameters[k],q[i])
#            aux_par_num = k+1  
#            for i in np.arange(int(self.n_qubit/2)):
#                qc.cz(q[2*i+shift],q[(2*i+1+shift)%self.n_qubit])
#            #####################################
#            for i in np.arange(self.n_qubit):
#                k =i + aux_par_num
#                qc.ry(parameters[k],q[i])
#            aux_par_num = k+1    
#            shift+=1
#            shift%=2
#        qc.save_statevector() 
#        qc.measure(q[:self.n_qubit], c[:self.n_qubit])
#        return qc 
#    def exactHamcostfunction(self,parameters):
#        qc = self.build_qc(parameters)
#        result = backend.run(qc, shots = 1).result()
#        _counts = result.get_counts()
#        _data, _data_table, entropy = data_processing(_counts, self.n_qubit)
#        _fin_statevec  = result.get_statevector(qc)
#        _fin_reduced_state = quantum_info.partial_trace(_fin_statevec,
#                                                        list(np.arange(self.n_qubit,self.num_ini_qubit) )# remove quits n_qubits+1 ~ ini_qubits
#                                                       )
#
#        
#        Hl, _r_list = self.gen_Hl(_fin_reduced_state.num_qubits, 
#                                  _fin_reduced_state)
#        #print(Hl, _r_list)
#        
#        
#        m = _fin_reduced_state.num_qubits + 1 ## m = n+1 ! look the paper VQSE 
#        q_list = self.gen_qlist(m,_r_list) # large population * large q! (small E )
#        #print('q_list',q_list)
#        
#        #evallist = []
#        #for i in np.arange(2**_fin_reduced_state.num_qubits):
#        #    evallist.append(_fin_reduced_state.data[i,i].real)
#        #evallist = np.array(evallist)
#        #mevallist = evallist.copy()
#        #mevallist.sort() 
#        #mevallist = mevallist[::-1][:m] # m diaognal element from lagest one! 
#        #mindex_list = []
#        #for i in mevallist:
#        #    mindex_list.append(np.where(evallist ==i )[0].squeeze() )
#        Hg = 1
#        for idx, k in enumerate(self.m_indexlist):
#            diag_comp = _fin_reduced_state.data[k,k].real
#            cof = q_list[idx]
#            Hg -=  cof * diag_comp
#            #print(cof,diag_comp)
#            
#        #print(evallist)
#        #print(mindex_list)
#        #print('mevalist', mevallist)
#        #E1 = 1 - np.sum(np.array(_r_list)/2)
#        
#        #print(q_list)
#
#        cost_ftn = Hg * self.GLratio + Hl * (1 - self.GLratio)        
#        return cost_ftn 
#
#    def exactRenyicostfunction(self,parameters):
#        _alpha = self.alpha # added term for Renyi cost ftn 
#        qc = self.build_qc(parameters)
#        result = backend.run(qc, shots = 1).result()
#        _counts = result.get_counts()
#        _data, _data_table, entropy = data_processing(_counts, self.n_qubit)
#        _fin_statevec  = result.get_statevector(qc)
#        _fin_reduced_state = quantum_info.partial_trace(_fin_statevec,
#                                                       list(self.n_qubit + np.arange(self.n_qubit)))
#        test_sum = 0
#        cost_ftn = 0
#        for i in np.arange(2**_fin_reduced_state.num_qubits):
#            for j in np.arange(2**_fin_reduced_state.num_qubits):
#                if i==j:
#                    pii = _fin_reduced_state.data[i,i]
#                    pii = pii.real
#                    cost_ftn += pii ** _alpha
#        cost_ftn /= _alpha *(1-_alpha)
#        return cost_ftn
#    def exactcostfunction(self,parameters):
#        qc = self.build_qc(parameters)
#        result = backend.run(qc, shots = 1).result()
#        _counts = result.get_counts()
#        _data, _data_table, entropy = data_processing(_counts, self.n_qubit)
#        _fin_statevec  = result.get_statevector(qc)
#        _fin_reduced_state = quantum_info.partial_trace(_fin_statevec,
#                                                        list(np.arange(self.n_qubit,self.num_ini_qubit) )# remove quits n_qubits+1 ~ ini_qubits
#                                                       )
#        test_sum = 0
#        cost_ftn = 0
#        for i in np.arange(2**_fin_reduced_state.num_qubits):
#            for j in np.arange(2**_fin_reduced_state.num_qubits):
#                if i == j:
#                    pii = _fin_reduced_state.data[i,i]
#                    pii = pii.real
#                    test_sum += pii
#                    cost_ftn += - np.log( pii ) * pii
#        cost_ftn += np.log(test_sum)
#        return cost_ftn
#    def gen_qlist(self,m, r_list):
#        """
#        create qlist for Hg from r_list of Hl 
#        """
#        LocalHam =  qt.tensor([  qt.qeye(2) for j in np.arange(self.n_qubit )])
#        for i in np.arange(self.n_qubit ):
#            LocalHam += -  r_list[i] * qt.tensor([ (1/2) * qt.sigmaz() if j == i else qt.qeye(2) 
#                                                for j in np.arange(self.n_qubit )])
#        result = 1 - LocalHam.eigenenergies()[:m]
#        return result 
#    def gen_Hl(self,num_view_qubits, partial_state):
#        """
#        returns
#        result = Hl value
#        r_list = r value list, for cal of Hl and Hg
#        """
#        
#        Hl = 1.
#        traceoverlist = np.arange(num_view_qubits) 
#        #print(traceoverlist)
#        _updownlist = []
#        for k in np.arange(num_view_qubits):
#            l = list(traceoverlist)
#            l.remove(k)
#            #print(l)
#            _temp_state = quantum_info.partial_trace(partial_state,
#                                                     l# remove quits n_qubits+1 ~ ini_qubits
#                                                     )
#            assert _temp_state.num_qubits == 1 
#            _updownlist.append( (_temp_state.data[0,0].real, _temp_state.data[1,1].real) )
#        #print(_updownlist)
#        r_list = [] # build r list for coef of Sz 
#        for k in  np.arange(self.n_qubit):
#            val = self.r1 + self.VQSEdelta * (k)
#            r_list.append(val)
#        
#        for idx, rho in enumerate(_updownlist): # Sz evaluation
#            Hl+= - r_list[idx] * ( rho[0] - rho[1] )/2 # i added half factor 
#        #print(idx, 'idx!!', rho[0], 'population!!' )
#        return Hl, r_list 
#    def gen_idxlist(self, reduced_state):
#        _, _r_list = self.gen_Hl(reduced_state.num_qubits, 
#                                  reduced_state)
#        
#        m = reduced_state.num_qubits + 1 ## m = n+1 ! look the paper VQSE 
#        q_list = self.gen_qlist(m,_r_list) # large population * large q! (small E )
#        #print('q_list',q_list)
#        
#        evallist = []
#        for i in np.arange(2**reduced_state.num_qubits):
#            evallist.append(reduced_state.data[i,i].real)
#        evallist = np.array(evallist)
#        mevallist = evallist.copy()
#        mevallist.sort() 
#        mevallist = mevallist[::-1][:m] # m diaognal element from lagest one! 
#        mindex_list = []
#        for i in mevallist:
#            mindex_list.append(np.where(evallist ==i )[0].squeeze() )
#        return mindex_list, mevallist
    
def VQSE_trial(opt,seed, lr ):
    """
    input opt, seed, lr 
        seed for initial circuit parameters
        lr for circuit tuning 
    returns minE, exact_learning_curve, minlist, EP
              
    """
    np.random.seed(seed)
    ini_parameters = 2. * np.pi * np.random.uniform(size = opt.num_parameters)
    qc_parameters = ini_parameters
    save_parameters  = qc_parameters
    optimizer = GradientDescent(learning_rate = lr , maxiter = opt.maxiter) #ADAM(maxiter = 10)
    t2 = time.time()

    opt.GLratio = 0.
    VQSEClass = VQSE(opt)
    qc = VQSEClass.build_qc(ini_parameters)#.exactHamcostfunction(ini_parameters)

    result = backend.run(qc, shots = 1).result()
    temp_state = result.get_statevector(qc)
    reduced_state = quantum_info.partial_trace(temp_state,
                                                list(np.arange(opt.n_qubit,opt.num_ini_qubit) )# remove quits n_qubits+1 ~ ini_qubits
                                               )

    eigenvalue_list = np.linalg.eigh(reduced_state.data)[0][::-1]
    print(eigenvalue_list[:opt.n_qubit+1])

    opt.m_indexlist, largeproplist = VQSE(opt).gen_idxlist(reduced_state)
    VQSEClass = VQSE(opt)
    print('large proplist', largeproplist)

    minE = 100. 
    minlist = 0.
    exact_learning_curve = []
    exact_learning_curve.append(VQSEClass.exactHamcostfunction(qc_parameters))
    for QC_iter in np.arange(opt.circuit_repeats ):
        opt.GLratio = (QC_iter+ 1 )/opt.circuit_repeats
        VQSEClass = VQSE(opt)
        qc = VQSEClass.build_qc(qc_parameters)#.exactHamcostfunction(ini_parameters)

        result = backend.run(qc, shots = 1).result()
        temp_state = result.get_statevector(qc)
        reduced_state = quantum_info.partial_trace(temp_state,
                                                    list(np.arange(opt.n_qubit,opt.num_ini_qubit) )# remove quits n_qubits+1 ~ ini_qubits
                                                   )

        # update m_list ! 
        opt.m_indexlist, largeproplist = VQSE(opt).gen_idxlist(reduced_state)
        VQSEClass = VQSE(opt)
        result = optimizer.minimize(fun = VQSEClass.exactHamcostfunction, # minimize! 
                                    x0  = qc_parameters)
        qc_parameters = result.x
        opt.exactparameters = qc_parameters
        energy = VQSEClass.exactHamcostfunction(qc_parameters)
        EP = VQSEClass.exactcostfunction(qc_parameters)
        exact_learning_curve.append(energy)
        print('exact Energy' ,energy,
               'exact EP', EP, 
              'time: ', time.time() - t2 )
        print('large prop list ', largeproplist)
        #if energy < minE:
        #    minE = energy 
        #    minlist = largeproplist.copy()
    return energy, exact_learning_curve, largeproplist, EP


def QNEEP_trial(opt,seed, lr ):
    """
    input opt, seed, lr 
        seed for initial circuit parameters
        lr for circuit tuning 
    returns minEP, exact_learning_curve, minlist
              
    """
    np.random.seed(seed)
    ini_parameters = 2. * np.pi * np.random.uniform(size = opt.num_parameters)
    qc_parameters = ini_parameters
    save_parameters  = qc_parameters
    optimizer = GradientDescent(learning_rate = lr , maxiter = opt.maxiter) #ADAM(maxiter = 10)
    t2 = time.time()
    
    opt.GLratio = 0.
    VQSEClass = VQSE(opt)
    qc = VQSEClass.build_qc(ini_parameters)#.exactHamcostfunction(ini_parameters)

    result = backend.run(qc, shots = 1).result()
    temp_state = result.get_statevector(qc)
    reduced_state = quantum_info.partial_trace(temp_state,
                                                list(np.arange(opt.n_qubit,opt.num_ini_qubit) )# remove quits n_qubits+1 ~ ini_qubits
                                               )

    eigenvalue_list = np.linalg.eigh(reduced_state.data)[0][::-1]
    print(eigenvalue_list[:opt.n_qubit+1])

    opt.m_indexlist, largeproplist = VQSE(opt).gen_idxlist(reduced_state)
    VQSEClass = VQSE(opt)
    print('large proplist', largeproplist)

    minEP = 100. 
    minlist = 0.
    exact_learning_curve = []
    exact_learning_curve.append(VQSEClass.exactcostfunction(qc_parameters))
    for QC_iter in np.arange(opt.circuit_repeats ):
        opt.GLratio = (QC_iter+ 1 )/opt.circuit_repeats
        VQSEClass = VQSE(opt)
        qc = VQSEClass.build_qc(qc_parameters)#.exactHamcostfunction(ini_parameters)

        result = backend.run(qc, shots = 1).result()
        temp_state = result.get_statevector(qc)
        reduced_state = quantum_info.partial_trace(temp_state,
                                                    list(np.arange(opt.n_qubit,opt.num_ini_qubit) )# remove quits n_qubits+1 ~ ini_qubits
                                                   )

        # update m_list ! 
        opt.m_indexlist, largeproplist = VQSE(opt).gen_idxlist(reduced_state)
        VQSEClass = VQSE(opt)
        result = optimizer.minimize(fun = VQSEClass.exactcostfunction, # minimize! 
                                    x0  = qc_parameters)
        qc_parameters = result.x
        opt.exactparameters = qc_parameters
        EP = VQSEClass.exactcostfunction(qc_parameters)
        exact_learning_curve.append(EP)
        print( 'exact EP', EP, 
              'time: ', time.time() - t2 )
        print('large prop list ', largeproplist)
        if EP < minEP:
            minEP = EP 
            minlist = largeproplist.copy()
    return minEP, exact_learning_curve, minlist