import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from scipy import optimize

from obj import PyTorchObjective

from tqdm import tqdm


    # make module executing the experiment
class Objective(nn.Module):
    def __init__(self, loss, output):
        super(Objective, self).__init__()
        self.loss = loss
        self.output = output
        self.labels = labels

        

    def forward(self):
        # output = self.linear(self.X)
        return self.loss(output, labels)

objective = Objective()
    
maxiter = 100

bounds = [] #need to fill this out
with tqdm(total=maxiter) as pbar:
    def verbose(xk):
        pbar.update(1)
    # try to optimize that function with scipy
    obj = PyTorchObjective(objective)
    xL = optimize.differential_evolution(obj.fun, bounds, callback=verbose)
    #xL = optimize.minimize(obj.fun, obj.x0, method='CG', jac=obj.jac)# , options={'gtol': 1e-2})