[![Paper](http://img.shields.io/badge/paper-arxiv.2307.13511-B31B1B.svg)](https://arxiv.org/abs/2307.13511)


# QNEE
QNEE is algorithm for estimating quantum entropies using neural network and quantum circuit. See arXiv 2307.13511 (https://arxiv.org/abs/2307.13511).

## Requirement 

Required libraries are written in requirements.txt.

[NOTES]

In ExecutionFiles folder, there is the python file used to produce QNEE result. 
In notebooks foler, "ReproducingVQSE-finite sampling" jupyter notebook file is a code for producing VQSE data. In the same folder, "Combine_figure_fnite_XXZ_VQSE_QNNEP" and "Combine_figure_fnite_XXZ_VQSE_QNNEP_4qubits" jupyter notebook files are codes for plotting main figure in our paper.
There are the data we used for QNEE paper in "VNEM\notebooks\data\arXiv".


This code utilizes Qiskit and a simulator of IBMQ (https://quantum-computing.ibm.com/). You might need your account. Run this. 

from qiskit import IBMQ               
IBMQ.save_account('your account')     


## Command line Example 
```
python -u VNestimator_train_strategy_three_XXZ_with_mk4circuit.py --n-shots 30000 --n-qubit 4 --num-ini-qubit 8 --directory C:/Users/sulee/Dropbox/Research/VN_ent_estimator_share/notebooks/data/QNEE/ --n-layer 3 --reps 10 --circuit_seed 4 --delta 0.05 --mag 3.0 --batch-size 30000 --lr 1e-05 --wd 5e-05 --qlr 0.01 --n-hidden 256 --n-layer 3 --n-train-repeats 20 --circuit-repeats 30 --ini_nn_iter 10000 --inter_nn_iter 100 --maxiter 25 --seed 0 
```
- Change the directory after --directory   
- Note that --n-train-repeats is a dummy parameter. This parameter is used in old version. Please ignore it. 
- To reproduce the QNEE result in QNEE paper, please execute the above code. Simple explanations about parameters are commented in VNestimator_train_strategy_three_XXZ_with_mk4circuit.py
  - With one A40 GPU card, it tooks 66 hours. Decrease circuit-repeats parameter for saving your time or use many GPUs.
- After run the command line, training with exact cost function is executed. Then, the neural network and quantum circuit are initialized and training with cost function from quantum circuit's data based on will be executed!

## Author

Sangyun Lee, Hyukjoon Kwon and Jae Sung Lee

