import sys 
sys.path.append('..')
from qiskit import IBMQ
#### IBM account might be necessary 
# please type your account 
#IBMQ.save_account('account')



# math tools
import numpy as np 
from scipy import stats 
import os
import copy

# class
import argparse
from argparse import Namespace

# neural net 
import torch 
import torch.nn as nn
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from neural.net import VonNeumannEP, train, validate


## xxzchain 
from xxzchain.xxzchain import groundXXZ, entangleandvislist, xxzansatz

# quantum circuit 
from qiskit import Aer
backend = Aer.get_backend("aer_simulator") # set simulator
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.algorithms.optimizers import COBYLA, ADAM, SLSQP, SPSA, GradientDescent
from qiskit.quantum_info import random_statevector, random_density_matrix

## custom lib 
from cirquit.tools import data_processing, initial_entangle_EP
from cirquit.CircuitModelMk4 import HybridCircuit # mark4 6/13 !! 

#from qiskit import tools
from qiskit import quantum_info

# graph tool 
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# date for mkdir 
from datetime import date

import pickle
import time

def save_obj(obj, name ):
    """
    inputs
    obj 
    name string
    """
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)



def main(opt):
    opt.n_clbit = opt.n_qubit
    opt.model = VonNeumannEP(opt).to(opt.device) # 이 부분을 distributed model? 
    opt.optim = torch.optim.Adam(opt.model.parameters(), opt.lr, 
                                 weight_decay=opt.wd)
    print(vars(opt))
    print("=" * 80)
    print("Make initial state of quantum state!")
    print("is cuda available?", torch.cuda.is_available() )

    opt.ini_state = groundXXZ(opt.num_ini_qubit, opt.delta,opt.mag).full().flatten()
    np.random.seed(opt.circuit_seed)
    ini_parameters = np.random.uniform(size = opt.num_parameters)
    qc_parameters = ini_parameters
    save_parameters  = ini_parameters
    
    reduced_state1 = quantum_info.partial_trace(opt.ini_state,
                                           list(np.arange(opt.n_qubit,opt.num_ini_qubit) )# remove quits n_qubits+1 ~ ini_qubits
                                           )
    reduced_state2 = quantum_info.partial_trace(opt.ini_state,
                                           list(np.arange(opt.n_qubit) )# remove quits n_qubits+1 ~ ini_qubits
                                           )
    opt.entangle_EP  = quantum_info.entropy(reduced_state1, base = np.e) 
    entangle_EP_ver2 = quantum_info.entropy(reduced_state2, base = np.e) 
    #get_ent(psiout,alpha = 1) # always half coarse-grained Entanglement EP ;;  log 2?  log2랑 비교하고 있었음..
    
    opt.eigenvalues = np.linalg.eigh(reduced_state1.data)[0][::-1] # added 6/14
    
    # make test data 
    print("make data for later test and training also quantum instance")
    QuantuClass = HybridCircuit(opt)
    qc = QuantuClass.build_qc(qc_parameters)
    counts = backend.run(qc).result().get_counts() # run quantum circuit 
    test_data, test_data_table, entrop  = data_processing(counts,opt.n_qubit) # data processed for ep calculation !
    
    print('test plug-in entropy: ',entrop)
    # make train  data 
    counts = backend.run(qc).result().get_counts()
    data, data_table, entrop  = data_processing(counts,opt.n_qubit)
    
    print("initial plug-in answer: ",entrop)
    
    print('Exact Entanglement entropy with base e:', opt.entangle_EP )
    print('Exact Entanglement entropy for cross check!:', entangle_EP_ver2 )
    opt.earlyexactcostftnval = QuantuClass.exactcostfunction(qc_parameters) # added 6/14 2023
    print("for comparision exact cost ftn training!", 
           opt.earlyexactcostftnval)
    
    optimizer = GradientDescent(learning_rate =  opt.qlr , maxiter = opt.maxiter) 
    
    t2 = time.time()
    learning_curve = []
    exact_learning = []
    for QC_iter in np.arange(opt.circuit_repeats):
        result = optimizer.minimize(fun = QuantuClass.exactcostfunction,
                                    x0  = qc_parameters)
        qc_parameters = result.x
        NEEP_output = QuantuClass.evaluationQC(qc_parameters)
        learning_curve.append(NEEP_output)
        exact_learning.append(QuantuClass.exactcostfunction(qc_parameters))
        
        print('qc neural net result', learning_curve[-1],
              'exact EP' , exact_learning[-1],
              'time: ', time.time() - t2 )
        
    opt.exact_learning = exact_learning
    
    print("revert initial param!" )
    qc_parameters  = ini_parameters
    print("initialized entanglement ep", 
          QuantuClass.exactcostfunction(qc_parameters) )
    
    print("=" * 80)
    print("Start initial training!")
    

    print("is cuda available?", torch.cuda.is_available() )
    # neural net
    #optim = torch.optim.Adam(opt.model.parameters(), opt.lr, weight_decay=opt.wd)

    # Data loader ! -> distributed data loader! 
    train_dataloader = DataLoader(data, batch_size=opt.batch_size, shuffle=True)
    test_dataloader  = DataLoader(test_data, batch_size=opt.n_shots, shuffle=False)

    loss_list = []
    test_loss_list = []
    normal_cond_list = []
    
    temp_loss, temp_normal = train(opt, opt.model,data_table,  
                                   opt.optim, train_dataloader)
    test_cost, _ = validate(opt, opt.model, 
                            test_data_table,
                            test_dataloader)
    
    opt.save_NEEP_output = test_cost
    opt.saveAdamState = copy.deepcopy(opt.optim.state_dict())
    opt.saveNNState   = copy.deepcopy(opt.model.state_dict())
    record_value = test_cost
    for i in np.arange(1, opt.ini_nn_iter ):
        temp_loss, temp_normal = train(opt, opt.model, data_table,  
                                      opt.optim, train_dataloader)
        test_cost, _ = validate(opt, opt.model, 
                            test_data_table,
                            test_dataloader)
        
        ## find nice optimal point!
        if test_cost[0] < opt.save_NEEP_output:
            print('save output: ', test_cost)
            opt.save_NEEP_output = copy.deepcopy(test_cost)
            opt.saveAdamState = copy.deepcopy(opt.optim.state_dict())
            opt.saveNNState   = copy.deepcopy(opt.model.state_dict())
            
            
        ## recording 
        if i%opt.record_freq == 0:
            loss_list.append(temp_loss)
            test_loss_list.append(test_cost.to('cpu').numpy())
            normal_cond_list.append(temp_normal)
            if record_value > test_cost:
                record_value = test_cost
            else:
                break  
                
    #opt.save_NEEP_output = loss_list[-1]
    opt.ini_loss_list = loss_list
    opt.ini_normal_cond_list = normal_cond_list
    opt.ini_eigenval = QuantuClass.estimateeigenval(ini_parameters)
    print("initial training end!")
    print("load trained neural network!")
    
    opt.model.load_state_dict(opt.saveNNState)
    opt.optim.load_state_dict(opt.saveAdamState)
    
    print("=" * 80)
    print("Set intermediate training! strategy 3 every time training !!!")
    print("is cuda available?", torch.cuda.is_available() )
    
    optimizer = GradientDescent(learning_rate =  opt.qlr , maxiter = opt.maxiter)

    opt.intermediate_estimated_EP = []
    opt.intermediate_estimated_NEEP = []
    reoptim_NEEP = []
    
    print("Start intermediate training!")
    t4 = time.time()
    
    inter_ent_list = []
    inter_exact_list = []
    inter_qc_parameters_list = [] # added 6/14
    inter_eigenvals_list = [] # added 6/14
    #QuantuClass = HybridCircuit(opt) # update save_NEEP_output...
    for QC_iter in np.arange(opt.circuit_repeats):
        result = optimizer.minimize(fun = QuantuClass.costfunction,
                                    x0 = qc_parameters)
        qc_parameters = result.x
        NEEP_output = QuantuClass.evaluationQC(qc_parameters)
        inter_ent_list.append(NEEP_output)
        inter_exact_list.append(QuantuClass.exactcostfunction(qc_parameters))
        inter_qc_parameters_list.append(qc_parameters) # added 6/14
        inter_eigenvals_list.append(QuantuClass.estimateeigenval(qc_parameters)) # added 6/14
        
        opt.save_NEEP_output = copy.deepcopy(QuantuClass.save_NEEP_output)
        opt.saveAdamState = copy.deepcopy(QuantuClass.optim.state_dict())
        opt.saveNNState   = copy.deepcopy(QuantuClass.model.state_dict())
        
        opt.inter_qc_parameters_list = inter_qc_parameters_list
        opt.inter_ent_list = inter_ent_list
        opt.inter_exact_list = inter_exact_list
        opt.inter_eigenvals_list = inter_eigenvals_list # added 6/16
        print('qc opt result', inter_ent_list[-1],
              'exact EP' , inter_exact_list[-1],
              'time: ', time.time() - t4 )
        os.makedirs(opt.directory, exist_ok = True)
        save_obj(opt, opt.directory + '/XXZtotal{num_ini_qubit}qubits_view{n_qubit}qubits_interiter{inter_nn_iter}_nshots{n_shots}_qlr{qlr}_cseed{cseed}'.format(
                                        num_ini_qubit = opt.num_ini_qubit, 
                                        n_qubit = opt.n_qubit,
                                        cseed = opt.circuit_seed,
                                        inter_nn_iter = opt.inter_nn_iter,
                                        n_shots = opt.n_shots,
                                        qlr = opt.qlr)       
                )
    
if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(
        description="Von Neumann Entropy Production Estimator for MIPT state"
    )
    parser.add_argument(
        "--n-shots", type=int, default=50000, help="the number of shots of quantum circuit"
    )
    parser.add_argument(
        "--n-qubit",
        type = int,
        default = 3,
        help="the number of qubits after coarse graining!",
    )
    parser.add_argument(
        "--reps",
        default = 5,
        type = int,
        help="the number of depth, must be largert than the # of qubits!)",
    )
    parser.add_argument(
        "--circuit_seed",
        default = 10,
        type = int,
        help="seed of quantum circuit",
    )
    parser.add_argument(
        "--delta",
        default = 0.05,
        type = float,
        help="xxz chain delta ",
    )
    parser.add_argument(
        "--mag",
        default = 0.1,
        type = float,
        help="xxz chain mag ",
    )
    parser.add_argument(
        "--num-ini-qubit",
        default = 6,
        type = int,
        help="the number of qubits before corasegraining, 8 default ",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        metavar="N",
        help="input batch size for training (default: 10000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate (default: 1e-5)",
    )
    parser.add_argument(
        "--qlr",
        type=float,
        default=0.2,
        help="quantum circuit learning rate (default: 0.2)",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=5e-5,
        help="learning rate (default: 5e-5)",
    )
    parser.add_argument(
        "--n-train-repeats",
        type=int,
        default=2,
        help="Repeat number of neural net train (default: 2)",
    )
    parser.add_argument(
        "--circuit-repeats",
        type=int,
        default=5,
        help="Repeat number of circuit optimization; total_repeats = n_train_repeat * circuit train_repeats (default: 5)",
    )
    parser.add_argument(
        "--ini_nn_iter",
        type=int,
        default=10000,
        help="number of iteration to initial train (default: 10000)",
    )
    parser.add_argument(
        "--inter_nn_iter",
        type=int,
        default=10000,
        help="number of iteration to intermediate train (default: 10000)",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=10,
        help="number of max iteration for quantum circuit optimizer (default: 10)",
    )
    parser.add_argument(
        "--n-hidden",
        type=int,
        default=128,
        help="number of hidden neuron (default: 128)",
    )
    parser.add_argument(
        "--n-layer",
        type=int,
        default=3,
        help="number of layer (default: 3)",
    )
    parser.add_argument(
        "--record-freq",
        type=int,
        default=100,
        metavar="N",
        help="number of iteration to train (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=398,  help="random seed (default: 398)"
    )
    parser.add_argument(
        "--directory", type=str, default='../notebooks/data/test',  help="directory"
    )
    parser.add_argument(
        "--device", type=str, default='cuda',  help="directory"
    )
    opt = parser.parse_args()
    opt.datasize = opt.n_shots
    opt.n_clbit = opt.n_qubit
    opt.num_ini_clbit = opt.num_ini_qubit  
    opt.state_dim = 2 ** opt.num_ini_qubit
    opt.n_token = int( 2 ** opt.n_qubit )
    opt.model = VonNeumannEP(opt).to(opt.device) 
    opt.num_parameters = int( 2 * opt.n_qubit *(opt.reps+1) ) # ? wrong? 
    
    # make directory 
    if not os.path.exists(opt.directory):
        os.makedirs(opt.directory)
    t0 = time.time()
    main(opt)
    print('Duration is', (time.time() - t0)/3600., 'hours' )
