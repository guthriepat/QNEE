import torch
import torch.nn as nn


class VonNeumannEP(nn.Module):
    def __init__(self, opt):
        """
        neural network for VonNeumannEP
        input must be integer not float or double 
        
        Parameters : opt : Class 
                        opt has parameters for learning
                            opt.n_token : # of input types (if data is 0 or 1 then n_token = 2 )
                            opt.n_hidden: # of nodes 
                            opt.n_layer : # of layer
                        s: input string or int 
        """
        super(VonNeumannEP, self).__init__()
        self.encoder = nn.Embedding(opt.n_token, opt.n_hidden)
        # input is a vector for a present state
        self.h = nn.Sequential()
        for i in range(opt.n_layer):
            self.h.add_module(
                "fc%d" % (i + 1), nn.Linear( opt.n_hidden,opt.n_hidden)
            )
            self.h.add_module("relu%d" % (i + 1), nn.ReLU())
        self.h.add_module("out", nn.Linear( opt.n_hidden, 1))
        
    def forward(self, s):
        s = self.encoder(s)
        return self.h(s)
    
def train(opt, model, data_table, optim, sampler): #, data, aux_data):
    """
    train neural net with Donsker--Varadhan style cost function
    """
    model.train()
    batch = next(iter(sampler)).int().to(opt.device) # send to int for embedding
    #print(batch.type())
    estimator = model(batch)
    optim.zero_grad() # initialize 
    d = opt.n_token
    normal_cond = 0
    for i in data_table:
        i = torch.tensor(i).to(opt.device)
        #print(i, type(i))
        normal_cond += torch.exp(model(i))
    # evaluate loss function !!
    loss = normal_cond  - 1 + (- estimator).mean()
    #loss = normal_cond - 1 + (- estimator).mean()
    loss.backward()
    optim.step()
    
    return loss.item(), normal_cond

def validate(opt, model,  data_table, sampler):
    """
    validate neural net!
    outputs
     loss : cost function value
     normal: nomal cond. 1 means sound trainning ! 
    """
    model.eval()

    with torch.no_grad():
        batch = next(iter(sampler)).int().to(opt.device)
        estimator = model(batch)
        normal_cond = 0
        for i in data_table:
            #print(i, type(i))
            i = torch.tensor(i).to(opt.device)
            #
            normal_cond += torch.exp(model(i))
            #print(model(i))
        normal = normal_cond 
        loss = normal_cond  - 1 + (- estimator).mean()
        
        
    return loss, normal 
