#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import sys
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from timeit import default_timer as timer
from tqdm import tqdm
from collections import defaultdict,Counter


import scipy.io
import optuna
import matplotlib.pyplot as plt

import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec

import scipy as sp
import matplotlib.pyplot as plt
os.chdir('/home/jovyan/work/VorkutovDA/')
from Data.DPDBlocks.blocks import AFIR, Delay, Prod_cmp, ABS, Polynomial
from Data.utilits.calculate_functions import ACPR_calc, MSE, NMSE
from torch.utils.data import DataLoader





# ## Dataload 

# In[9]:


Batch_size = 64
name = 'Data/BlackBoxData/BlackBoxData_80'
# name = 'BlackBoxData'
# name = '../BlackBoxData/data1'
mat = scipy.io.loadmat(name)
x = np.array(mat['x']).reshape(-1,1)/2**15
d = np.array(mat['y']).reshape(-1,1)/2**15
# x = np.array(mat['xE']).reshape(-1,1)/2**15
# d = np.array(mat['d']).reshape(-1,1)/2**15
# x, d = mat['xE'], mat['d']
x_real, x_imag = torch.from_numpy(np.real(x)), torch.from_numpy(np.imag(x))
d_real, d_imag = torch.from_numpy(np.real(d)), torch.from_numpy(np.imag(d))
X = torch.DoubleTensor(torch.cat((x_real, x_imag), dim=1)).reshape(-1,2,1)
D = torch.DoubleTensor(torch.cat((d_real, d_imag), dim=1)).reshape(-1,2,1)

train_queue = torch.utils.data.DataLoader(
    torch.cat((X,D),dim=-1), batch_size=Batch_size)#, pin_memory=True)

valid_queue = torch.utils.data.DataLoader(
    torch.cat((X,D),dim=-1), batch_size=X.shape[0])#,pin_memory=True)


# In[10]:


#NewDAtaLoader


# ## Train Function

# In[11]:


loss_fn = nn.MSELoss()


# ## Class Model

# In[12]:


import os
import sys
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from timeit import default_timer as timer
#from tqdm import tqdm_notebook
from tqdm.notebook import trange, tqdm
from collections import defaultdict,Counter
import pickle

import scipy.io

from Data.DPDBlocks.blocks import AFIR,ABS,Polynomial,Delay,Prod_cmp

import matplotlib.pyplot as plt

import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec


def eval_model(valid_queue, model, criterion):
    for step, (valid) in enumerate(valid_queue):
        model.eval()
        input_batch = Variable(valid[:,:,:1],requires_grad=False).permute(2,1,0).cpu()
        desired = Variable(valid[:,:,1:],requires_grad=False).permute(2,1,0).cpu()
        #out = model.forward(input_batch)
        out = sum(list(map( lambda n: n(input_batch),  model)))

        loss=criterion(out,desired)
        #draw_spectrum(input_batch,desired,out)
        accuracy = NMSE(input_batch, out - desired)

    return loss,accuracy

def train_of_epoch(train_queue, model, criterion, optimizer):
    

    
    for step, (train) in enumerate(train_queue):

        input_batch = Variable(train[:,:,:1],requires_grad=False).permute(2,1,0).cpu()
        desired = Variable(train[:,:,1:],requires_grad=False).permute(2,1,0).cpu()
        optimizer.zero_grad()
        #out = model.forward(input_batch)
        out = sum(list(map( lambda n: n(input_batch),  model)))

        loss = criterion(out, desired)
        
        
        loss.backward()
        
        optimizer.step()


def train(train_queue, valid_queue, model, criterion, optimizer,n_epoch,
          scheduler,log_every=1,save_flag=True,path_to_experiment='', dataFromEpohAccuracy = [], dataFromEpohLoss = []):
    
    
    model.train()

    min_loss=0
    hist=defaultdict(list)
    t0=timer()
    for it in tqdm (range(n_epoch)):
        model.train(True)
        train_of_epoch(train_queue, model, criterion, optimizer)
        scheduler.step()
        if it%log_every==0:
            loss_v,accuracy_v=eval_model(valid_queue, model, loss_fn)
            print('Loss = ',loss_v.cpu().detach().numpy(), 'Accuracy = ', accuracy_v.cpu().detach().numpy(), 'dbs')
            dataFromEpohAccuracy.append(accuracy_v)
            dataFromEpohLoss.append(loss_v)
            
            if save_flag:
                with open(path_to_experiment + '/hist.pkl', 'wb') as output:
                    pickle.dump(hist, output)

                    torch.save(model.state_dict(), path_to_experiment + '/model.pt')
                if hist['train_loss_db'][-1] < min_loss:
                            min_loss = hist['train_loss_db'][-1]
                            torch.save(model.state_dict(), path_to_experiment + '/best_model.pth')

#class Cell_try_2(nn.Module):
#    def __init__(self,M=15,D=0,Poly_order=8,Passthrough=False):
#        super(Cell_try_2,self).__init__()
#        self.f = AFIR(M,0)
#        self.pol = Polynomial(Poly_order,Passthrough)
#        self.prod = Prod_cmp()
#        self.delay = Delay(D)
#    def forward(self,x):
#        #return self.prod(self.f(self.delay(x)), self.pol(self.delay(x)))
#        return self.prod( self.f(x), self.pol(self.f(self.delay(x))) )
class Cell_try_2(nn.Module):
    def __init__(self,M=15,D=0,Poly_order=8,Passthrough=False):
        super(Cell_try_2,self).__init__()
        self.f = AFIR(M,0)
        self.pol = Polynomial(Poly_order,Passthrough)
        self.prod = Prod_cmp()
        self.delay = Delay(D)
    def forward(self,x):
        #return self.prod(self.f(self.delay(x)), self.pol(self.delay(x)))
        return self.prod( self.f(self.delay(x)), self.f(self.pol(self.delay(x))) )


# ### Тут считаем самую большую модель по NMSE

# In[13]:


D = {'p': [4, 5, 6, 7, 8, 9], 'k' : [3, 5, 7, 9], 'z' : [-2, -1, 0, 1, 2]}
ex_D = {} # extremum vals of D
step_size=10
gamma=0.1
for key in D.keys():
  ex_D[key] = [ min(D[key]), max(D[key])]

# complex reference model
ref_model = {'k': [9,5,9,7,9],'p': [9,8,6,9,6]}

### params of the functional
score_huge = -37
score_min = -20.0

accuracHuge = []
Loss_cur = []

complex_huge = 2 * ( sum(ref_model['k']) + sum(ref_model['p']) )
complex_min = 2 * (4 * 5 + 3 * 5)
trtr_coef = 0.4
PATH = "./Program/ModelGrid"
def objective(trial):
  # create and train NN
  net = torch.nn.ModuleList()

  complex_cur = 0

  for i in range(5):
    # det hyperparams 
    poly_ord = trial.suggest_int('p'+str(i), ex_D['p'][0], ex_D['p'][1])
    conv_ord = trial.suggest_int('k'+str(i), ex_D['k'][0], ex_D['k'][1], step=2)
    net.append(Cell_try_2(M=conv_ord, D=(ex_D['z'][0] + 1*i), Poly_order=poly_ord))
    complex_cur = complex_cur + poly_ord + conv_ord

  net = net.to(torch.device('cpu'))
  optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
  train(train_queue, valid_queue, net, loss_fn, optimizer, 150, scheduler, save_flag=False)

  loss_cur, accuracy_cur = eval_model(valid_queue, net, loss_fn)
  score_cur = accuracy_cur.item()
  accuracHuge.append(score_cur)
  Loss_cur.append(loss_cur.item() )
  #torch.save(net,PATH)
  return score_cur
  #return  (complex_cur - complex_min) / (complex_huge - complex_min) + trtr_coef * (score_huge - score_cur) / (score_huge - score_min)


# In[14]:


study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction="minimize", pruner=optuna.pruners.HyperbandPruner())
study.optimize(objective, n_trials=100)
print(study.best_params)


# In[ ]:


with open('./Program/AccurasiNMSEGrid', 'wb') as fp:
    pickle.dump(score_cur, fp)
with open('./Program/LossGridNMSE', 'wb') as fp:
    pickle.dump(Loss_cur, fp)


# In[13]:


with open('./Program/BestPAramGridNMSE', 'wb') as fp:
    pickle.dump(study.best_params, fp)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




