#!/usr/bin/env python
# coding: utf-8

# In[ ]:


print("printing")


# In[ ]:


print("Starting Experiments!")

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist

import torch_geometric
import torch_geometric.nn as gnn

from torch_geometric.datasets import QM9
import GCL.augmentors
import GCL.augmentors as A

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression, LinearRegression
import itertools
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

import GCL.augmentors as A
import edge_removing as A_alternate
from GCL.augmentors import node_dropping, ppr_diffusion, feature_dropout, edge_adding, rw_sampling
import matplotlib.pyplot as plt

import math
import pandas as pd


import timeit
import os
from datetime import datetime

from rdkit.Chem import PeriodicTable
from rdkit import Chem

device = 'cuda' if torch.cuda.is_available else 'cpu'


# In[ ]:


#record start time
t_0 = timeit.default_timer()
# call function


# In[ ]:


parameters = {}
parameters['batch_size'] = 4096


# In[ ]:


#QM9 dataset list of input features and list of target features
dataset = "QM9"

#list of input features in QM9 dataset
x_index = {0: 'H atom?',
1: 'C atom?',
2: 'N atom?',
3: 'O atom?',
4: 'F atom?',
5: 'atomic_number',
6: 'aromatic',
7: 'sp1',
8: 'sp2',
9: 'sp3',
10: 'num_hs'}
x_index_list = ['H atom?', 
                'C atom?', 
                'N atom?', 
                'O atom?', 
                'F atom?', 
                'atomic_number', 'aromatic', 
                'sp1',
                'sp2',
                'sp3',
                'num_hs']


#list of target features in QM9 dataset
qm9_index = {0: 'Dipole_moment',
1: 'Isotropic_polarizability',
2: 'HOMO',
3: 'LUMO',
4: 'HOMO_LUMO_gap',
5: 'Electronic_spatial_extent',
6: 'Zero_point_vibrational_energy',
7: 'Internal_energy_at_0K',
8: 'Internal_energy_at_298.15K',
9: 'Enthalpy_at_298.15K',
10: 'Free_energy_at_298.15K',
11: 'Heat_capacity_at_298.15K',
12: 'Atomization_energy_at_0K',
13: 'Atomization_energy_at_298.15K',
14: 'Atomization_enthalpy_at_298.15K',
15: 'Atomization_free_energy_at_298.15K',
16: 'Rotational_constant_A',
17: 'Rotational_constant_B',
18: 'Rotational_constant_C'}


# In[ ]:


#parameters for GCL methods

tr_batch_size = 1000
val_batch_size = 200
test_batch_size = 100
tr_ratio = 0.9
val_ratio = 0.09
test_ratio = 0.01
num_workers = 2
shuffle = True
qm9_index_list = ['Dipole_moment', 
                  'Isotropic_polarizability',
                  'Highest_occupied_molecular_orbital_energy',
                  'Lowest_unoccupied_molecular_orbital_energy',
                  'Gap_between_previous_2',
                  'Electronic_spatial_extent',
                  'Zero_point_vibrational_energy',
                  'Internal_energy_at_0K',
                  'Internal_energy_at_298.15K',
                  'Enthalpy_at_298.15K',
                  'Free_energy_at_298.15K',
                  'Heat_capacity_at_298.15K',
                  'Atomization_energy_at_0K',
                  'Atomization_energy_at_298.15K',
                  'Atomization_enthalpy_at_298.15K',
                  'Atomization_free_energy_at_298.15K',
                  'Rotational_constant_A',
                  'Rotational_constant_B',
                  'Rotational_constant_C']

parameters = {}
parameters['tr_batch_size'] = tr_batch_size

parameters_used = {}
parameters_used['dataset'] = dataset
parameters_used['tr_batch_size'] = tr_batch_size
parameters_used['val_batch_size'] = val_batch_size
parameters_used['test_batch_size'] = test_batch_size
parameters_used['tr_ratio'] = tr_ratio
parameters_used['val_ratio'] = val_ratio
parameters_used['test_ratio'] = test_ratio
parameters_used['num_workers'] = num_workers
parameters_used['shuffle'] = shuffle
parameters_used['target_properties'] = qm9_index_list

#vicreg loss function parameters
sim_coeff = 25
std_coeff = 25
cov_coeff = 1

#list of training augmentations
tr_augmentations = [#A.RWSampling(num_seeds=1000, walk_length=10),
                      A.NodeDropping(pn=0.1),
                      A.FeatureMasking(pf=0.1),
                      A_alternate.EdgeRemoving(pe=0.1)]

#list of validation augmentations
val_augmentations = []

#list of test augmentations
test_augmentations = []

#number of choices for augmentations for training, validation, and test sets, respectively
tr_num_choices = 1
val_num_choices = 0
test_num_choices = 0

#Adam parameters
Adam_learning_rate = 0.002
Adam_weight_decay = 5e-4
adam = {'lr': Adam_learning_rate, 
        'weight_decay': Adam_weight_decay
       }

#dictionary of optimizers used
optimizers = {}
optimizers['adam'] = adam



#dictionary of loss functions used
loss_functions_used = {}

vicreg = {'sim_coeff': sim_coeff,
          'std_coeff': std_coeff,
          'cov_coeff': cov_coeff
         }

loss_functions_used['vicreg'] = vicreg

augmentations_used = {}
augmentations_used['tr_augmentations'] = tr_augmentations
augmentations_used['val_augmentations'] = val_augmentations
augmentations_used['test_augmentations'] = test_augmentations
augmentations_used['tr_num_choices'] = tr_num_choices
augmentations_used['val_num_choices'] = val_num_choices
augmentations_used['test_num_choices'] = test_num_choices


periodic_table = Chem.GetPeriodicTable()


# In[ ]:


#parameters for downstream ML models

#dictionary of parameters used for downstream Linear Models
lm_parameters = {'default': 'used default for all parameters'}

#dictionary of parameters used for downstream Random Forest models
rf_parameters = {'n_estimators': 100, 
                 'max_depth': 10 }

#dictionary of parameters used for downstream LightGBM models
lgbm_params = {'boosting_type': 'gbdt',
               'objective': 'regression',
               'metric': {'l2', 'l1'},
               'num_leaves': 31,
               'learning_rate': 0.05,
               'force_col_wise': 'true',
               'feature_fraction': 0.9,
               'bagging_fraction': 0.8,
               'bagging_freq': 5,
               'verbose': -1
            }
lgbm_parameters = {'params': lgbm_params,
            'num_boost_round': 20,
            'callbacks': [lgb.early_stopping(stopping_rounds=5)]
                  }


downstream_model_parameters = {}
downstream_model_parameters['lm_parameters'] = lm_parameters
downstream_model_parameters['rf_parameters'] = rf_parameters
downstream_model_parameters['lgbm_parameters'] = lgbm_parameters



parameters_used['loss_functions_used'] = loss_functions_used
parameters_used['augmentations_used'] = augmentations_used
parameters_used['optimizer'] = optimizers
parameters_used['adam_learning_rate'] = Adam_learning_rate
parameters_used['adam_weight_decay'] = Adam_weight_decay
parameters_used['downstream_model_parameters'] = downstream_model_parameters



# In[ ]:





# In[ ]:





# In[ ]:


whole_dataset = QM9(root = 'data/')

xenonpy_tr_df = pd.read_csv('XenonPy_transformed_datasets/QM9/xenon_tr.csv')
xenonpy_val_df = pd.read_csv('XenonPy_transformed_datasets/QM9/xenon_val.csv')
xenonpy_test_df = pd.read_csv('XenonPy_transformed_datasets/QM9/xenon_test.csv')



n = whole_dataset.len()
tr_n = math.floor(tr_ratio*n) # Number of QM9 to use as training data
val_n = math.floor(val_ratio*n)
test_n = math.floor(test_ratio*n)


all_inds = range(n)
tr_inds, val_inds = train_test_split(all_inds, train_size = tr_n, random_state = 24)
val_test_inds = range(n - tr_n)
val_inds, test_inds = train_test_split(val_test_inds, train_size = val_n, random_state = 24)
#only run test set at very, very end
#keep random state as is, or else rerun XenonPy transformations
    #XenonPy files above were created using random_state = 24

train_sampler = torch.utils.data.SubsetRandomSampler(tr_inds)
val_sampler = torch.utils.data.SubsetRandomSampler(val_inds)
test_sampler = torch.utils.data.SubsetRandomSampler(test_inds)

# We need to make a train and validation set since QM9 does not provide them
train_set = torch.utils.data.Subset(whole_dataset, tr_inds)
val_set = torch.utils.data.Subset(whole_dataset, val_inds)
test_set = torch.utils.data.Subset(whole_dataset, test_inds)


train_loader = torch_geometric.loader.DataLoader(train_set, batch_size = tr_batch_size,
                                                shuffle = shuffle, num_workers = num_workers)
                                              

val_loader = torch_geometric.loader.DataLoader(val_set, batch_size=val_batch_size,
                                            shuffle=shuffle, num_workers=num_workers, drop_last=True)
                                            
test_loader = torch_geometric.loader.DataLoader(test_set, batch_size=test_batch_size,
                                            shuffle=shuffle, num_workers=num_workers)
                                      
    

big_train_loader = torch_geometric.loader.DataLoader(train_set, batch_size = int(1e9),
                                                shuffle = True, num_workers = 2,) #gets entire training set


big_val_loader = torch_geometric.loader.DataLoader(val_set, batch_size = int(1e9),
                                                shuffle = True, num_workers = 2,) #gets entire validation set


val_aug = A.RandomChoice([], num_choices = 0)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


qm9_index = {0: 'Dipole moment',
1: 'Isotropic polarizability',
2: 'Highest occupied molecular orbital energy',
3: 'Lowest unoccupied molecular orbital energy',
4: 'Gap between previous 2',
5: 'Electronic spatial extent',
6: 'Zero point vibrational energy',
7: 'Internal energy at 0K',
8: 'Internal energy at 298.15K',
9: 'Enthalpy at 298.15K',
10: 'Free energy at 298.15K',
11: 'Heat capacity at 298.15K',
12: 'Atomization energy at 0K',
13: 'Atomization energy at 298.15K',
14: 'Atomization enthalpy at 298.15K',
15: 'Atomization free energy at 298.15K',
16: 'Rotational constant A',
17: 'Rotational constant B',
18: 'Rotational constant C',}



# In[ ]:


from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.rep_dim = 128
        self.emb_dim = 256
        
        # Data under graph
        self.conv1 = GCNConv(whole_dataset.num_node_features, self.rep_dim // 2)
        self.bn1 = nn.BatchNorm1d(self.rep_dim // 2)
        self.a1 = nn.LeakyReLU(0.02)
        
        self.conv2 = GCNConv(self.rep_dim // 2, self.rep_dim) # To Rep Space
        self.bn2 = nn.BatchNorm1d(self.rep_dim)
        
        # Projection to representation
        self.mpool1 = gnn.global_mean_pool
        #self.fc1 = nn.Linear(self.rep_dim, self.rep_dim)
        
        # Graph 2
        self.conv3 = GCNConv(self.rep_dim, self.rep_dim * 2) # To Emb Space
        self.bn3 = nn.BatchNorm1d(self.rep_dim * 2)
        
        # Projection to embedding
        #self.mpool2 = gnn.global_mean_pool
        #self.fc2 = nn.Linear(self.emb_dim, self.emb_dim) # Linear to rep?
        
    def forward(self, data, binds):
        x = data[0].float().to(device)
        edge_index = data[1].to(device)
        
        x = self.conv1(x, edge_index)
        x = self.a1(self.bn1(x))
        x = F.dropout(x, training=self.training)
        
        x = self.bn2(self.conv2(x, edge_index))
        
        x_rep = self.mpool1(x, binds)
        
        x_emb = self.conv3(x, edge_index)
        return x_rep, x_emb



def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def VicRegLoss(x, y):
    # https://github.com/facebookresearch/vicreg/blob/4e12602fd495af83efd1631fbe82523e6db092e0/main_vicreg.py#L184
    # x, y are output of projector(backbone(x and y))
    
    # These are the default params used in natural image vicreg
    sim_coeff = 25
    std_coeff = 25
    cov_coeff = 1
    
    
    repr_loss = F.mse_loss(x, y)

    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)

    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

    cov_x = (x.T @ x) / (parameters['batch_size'] - 1)
    cov_y = (y.T @ y) / (parameters['batch_size'] - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
        x.shape[1]
    ) + off_diagonal(cov_y).pow_(2).sum().div(x.shape[1])
    
    # self.num_features -> rep_dim?
    loss = (
        sim_coeff * repr_loss
        + std_coeff * std_loss
        + cov_coeff * cov_loss
    )
    return loss


def train(parameters):
    
    device = 'cuda'

    model = GCN().to(device)
    n_epochs = 1

    
    row_ind = 0
    
    optimizer = torch.optim.Adam(model.parameters(), lr=Adam_learning_rate, weight_decay=Adam_weight_decay)
    transfer_mat = torch.zeros((len(qm9_index.keys()), 6))

    tr_losses = []
    val_losses = []

    for epoch in range(0,n_epochs+1):
        epoch_losses = []
        for batch in train_loader:
            optimizer.zero_grad()

            batch_inds = batch.batch.to(device)

            # batch of graphs has edge attribs, node attribs - (n_nodes, n_features+1) -> concat (n_nodes, attrib1)

            batch.x = batch.x.float()#.to(device)
            #batch.edge_index = batch.edge_index.to(device)
            
            # Barlow - get 2 random views of batch
            #print(batch.x, batch.edge_index, batch.edge_attr)
            #print(aug, type(aug))
            b1 = aug(batch.x, batch.edge_index, batch.edge_attr)
            b2 = aug(batch.x, batch.edge_index, batch.edge_attr)
            
            # Embed each batch (ignoring representations)
            r1, e1 = model(b1, batch_inds)
            r2, e2 = model(b2, batch_inds)

            loss = VicRegLoss(e1, e2)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.data.item())

        print('epoch train loss', sum(epoch_losses) / len(epoch_losses))
        tr_losses.append(sum(epoch_losses) / len(epoch_losses))

        if epoch % 4 == 0:

            # Downstream supervised loss
            
#             for batch in big_train_loader: # take entire train set
#                 with torch.no_grad():
#                     # Embed training set under model
#                     rep_tr, _ = model(val_aug(batch.x, batch.edge_index, batch.edge_attr), batch.batch.to(device))


#                     for val_batch in val_loader:
#                         # Embed validation set under model
#                         rep_val, _ = model(val_aug(val_batch.x, val_batch.edge_index, val_batch.edge_attr), val_batch.batch.to(device))

#                         # For each task in QM9
#                         for tar_ind in range(batch.y.shape[1]):
#                             # Fit a model on model representation of train set

#                             #print(rep_tr.shape, batch.y[tar_ind].shap)
#                             lm = LinearRegression().fit(rep_tr.cpu(), batch.y[:,tar_ind])
#                             # Test the model on model repersentation of val set
#                             tar_yhat = lm.predict(rep_val.cpu())
#                             mse_met = mean_squared_error(val_batch.y[:,tar_ind], tar_yhat).item()
#                             r2_met = r2_score(val_batch.y[:,tar_ind], tar_yhat)
#                             #print(qm9_index[tar_ind], mse_met, r2_met)
#                             transfer_mat[tar_ind, row_ind] = mse_met
#                         row_ind += 1

            # VicReg Validation Loss
            val_loss = []
            for batch in val_loader:
                with torch.no_grad():
                    # VicReg validation loss
                    b1 = aug(batch.x, batch.edge_index, batch.edge_attr)
                    b2 = aug(batch.x, batch.edge_index, batch.edge_attr)
                    r1, e1 = model(b1, batch.batch.to(device))
                    r2, e2 = model(b2, batch.batch.to(device))

                    val_loss.append(VicRegLoss(e1, e2).item())

            val_losses.append(torch.mean(torch.FloatTensor(val_loss)))

    #plt.plot(tr_losses)
    plt.plot(val_losses, label = parameters['aug_str'])
    
    return model, tr_losses, val_losses, transfer_mat

import os

def trymkdir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


# In[ ]:





# In[ ]:


# Here is sample code for how to implement an "ablation" of 2-at-a-time augmentations
import GCL.augmentors as A
from GCL.augmentors import node_dropping, ppr_diffusion, feature_dropout, edge_adding, rw_sampling

aug = A.RandomChoice([#A.RWSampling(num_seeds=1000, walk_length=10),
                      A.NodeDropping(pn=0.1),
                      A.FeatureMasking(pf=0.1),
                      A_alternate.EdgeRemoving(pe=0.1)],
                     num_choices=1)

print(aug, type(aug))

# From a set of augmentations of length n_augmentations
aug_set = [A.NodeDropping(pn=0.1), A.FeatureMasking(pf=0.1), A_alternate.EdgeRemoving(pe=0.1), A.EdgeAdding(pe=0.1)]
           #ppr_diffusion, feature_dropout, edge_adding, rw_sampling]
    #A.PPRDiffusion()
aug_strs = ['NodeDropping', 'FeatureMasking', 'EdgeRemoving', 'EdgeAdding']
print(aug_strs)

# First get all pairs of indexes on-off in a list of length n_augmentations
aug_inds = list(itertools.product([0, 1], repeat=len(aug_set)))
aug_inds = [x for x in aug_inds if sum(x)==1]
print(aug_inds)

# Then for each augmentation, train and test a VicReg model trained under that augment
parameters = {}
parameters['batch_size'] = 64
parameters['learning_rate'] = 0.002
# etc parameters here which define model, hparams

for aug_index in aug_inds:
    print("aug_index: ", aug_index)
    tr_augs = []
    tr_strs = []
    for ind, augi in enumerate(aug_index):
        if augi == 1:
            tr_augs.append(aug_set[ind])
            tr_strs.append(aug_strs[ind])
            
    print("tr_strs: ", tr_strs[0])

    trymkdir(f'aug_sweep3-4/{tr_strs[0]}')
    
    tr_aug = A.RandomChoice(tr_augs, num_choices = 1)
    
    #print(tr_aug, type(tr_aug))
    parameters['train_aug'] = tr_aug
    parameters['aug_str'] = tr_strs
    
    model, train_loss, val_loss, transfer_mat = train(parameters)
    #transfer_scores = transfer_score(model, parameters)
    
    #print(transfer_mat)
    torch.save(model.state_dict(), f'aug_sweep3-4/{tr_strs[0]}/model.pt')
    torch.save(train_loss,  f'aug_sweep3-4/{tr_strs[0]}/train_loss.pt')
    torch.save(val_loss,  f'aug_sweep3-4/{tr_strs[0]}/val_loss.pt')
    torch.save(transfer_mat,  f'aug_sweep3-4/{tr_strs[0]}/transfer_mat.pt')
    
plt.legend()
plt.show()
    


# In[ ]:





# In[ ]:


# Here is sample code for how to implement an "ablation" of 2-at-a-time augmentations


# From a set of augmentations of length n_augmentations
aug_set = [A.NodeDropping(pn=0.1), A.FeatureMasking(pf=0.1), A_alternate.EdgeRemoving(pe=0.1), A.EdgeAdding(pe=0.1)]
           #ppr_diffusion, feature_dropout, edge_adding, rw_sampling]
    #A.PPRDiffusion()
aug_strs = ['NodeDropping', 'FeatureMasking', 'EdgeRemoving', 'EdgeAdding']
print(aug_strs)

# First get all pairs of indexes on-off in a list of length n_augmentations
aug_inds = list(itertools.product([0, 1], repeat=len(aug_set)))
aug_inds = [x for x in aug_inds if sum(x)==3]
print(aug_inds)

# Then for each augmentation, train and test a VicReg model trained under that augment
parameters = {}
parameters['batch_size'] = 64
parameters['learning_rate'] = 0.002

experiment = 'aug_sweep3-4'
trymkdir(experiment)
# etc parameters here which define model, hparams

for aug_index in aug_inds:

    tr_augs = []
    tr_strs = []
    for ind, augi in enumerate(aug_index):
        if augi == 1:
            tr_augs.append(aug_set[ind])
            tr_strs.append(aug_strs[ind])
            
    print(tr_strs)
    trymkdir(f'{experiment}/{tr_strs}')
    
    tr_aug = A.RandomChoice(tr_augs, num_choices = 1)
    
    #print(tr_aug, type(tr_aug))
    parameters['train_aug'] = tr_aug
    parameters['aug_str'] = tr_strs
    
    model, train_loss, val_loss, transfer_mat = train(parameters)
    #transfer_scores = transfer_score(model, parameters)
    
    #print(transfer_mat)
    torch.save(model.state_dict(), f'{experiment}/{tr_strs}/model.pt')
    torch.save(train_loss,  f'{experiment}/{tr_strs}/train_loss.pt')
    torch.save(val_loss,  f'{experiment}/{tr_strs}/val_loss.pt')
    torch.save(transfer_mat,  f'{experiment}/{tr_strs}/transfer_mat.pt')
    
plt.legend()
plt.show()


# In[ ]:





# In[ ]:


mse_scores = torch.zeros((19, len(aug_strs)))
for i_str, stri in enumerate(aug_strs):
    vec = torch.load(f'aug_sweep3-4/{stri}/val_loss.pt')
    
    model = GCN().to(device)
    model.load_state_dict(torch.load(f'aug_sweep3-4/{stri}/model.pt'))
    
    plt.plot(vec, label = stri)
    print(vec)
    
    for batch in big_train_loader: # take entire train set
        with torch.no_grad():
            # Embed training set under model
            rep_tr, _ = model(val_aug(batch.x, batch.edge_index, batch.edge_attr), batch.batch.to(device))
            if torch.cuda.is_available():
                rep_tr = rep_tr.to("cpu")
            rep_tr = pd.DataFrame(rep_tr.numpy())
            rep_tr.join(xenonpy_tr_df)

            val_tracker = 0
            print("One")
            
            for val_batch in val_loader:
                # Embed validation set under model
                rep_val, _ = model(val_aug(val_batch.x, val_batch.edge_index, val_batch.edge_attr), val_batch.batch.to(device))
                if torch.cuda.is_available():
                    rep_val = rep_val.to("cpu")
                rep_val = pd.DataFrame(rep_val.numpy())
                rep_val.join(xenonpy_val_df.iloc[val_tracker:(val_tracker+val_batch_size)])
                
                print("val_batch: ", val_batch)
                
                # For each task in QM9
                for tar_ind in range(batch.y.shape[1]):
                    # Fit a model on model representation of train set
                    print("tar_ind: ", tar_ind)
                    #print(rep_tr.shape, batch.y[tar_ind].shap)
                    lm = LinearRegression().fit(rep_tr, batch.y[:,tar_ind])
                    # Test the model on model repersentation of val set
                    tar_yhat = lm.predict(rep_val)
                    mse_met = mean_squared_error(val_batch.y[:,tar_ind], tar_yhat).item()
                    r2_met = r2_score(val_batch.y[:,tar_ind], tar_yhat)
                    #print(qm9_index[tar_ind], mse_met, r2_met)
                    mse_scores[tar_ind, i_str] = mse_met
        
                    
                val_tracker += val_batch_size
        print("Five")
plt.legend()
plt.show()


# In[ ]:





# In[ ]:


#print(mse_scores, mse_scores.shape)

# For one score, 

for i, row in enumerate(mse_scores):
    name = qm9_index[i]
    
    plt.bar(x = range(len(row)), height = row)
    plt.xticks(range(len(row)), aug_strs)
    plt.xlabel('Single Augmentation')
    plt.ylabel('Validation MSE Score')
    plt.title(name)
    plt.show()


# In[ ]:





# In[ ]:


# For all-but-one-augmentation
mse_scores = torch.zeros((19, len(aug_strs)))
rf_mse_scores = torch.zeros((19, len(aug_strs)))
lgb_mse_scores = torch.zeros((19, len(aug_strs)))
baseline_mse_scores = torch.zeros((19, len(aug_strs)))


#augs = os.walk('aug_sweep3-4')
print(next(os.walk('aug_sweep3-4'))[1])

for i_str, stri in enumerate(next(os.walk('aug_sweep3-4'))[1]):
    vec = torch.load(f'aug_sweep3-4/{stri}/val_loss.pt')
    
    model = GCN().to(device)
    model.load_state_dict(torch.load(f'aug_sweep3-4/{stri}/model.pt'))
    
    plt.plot(vec, label = stri)
    print(vec)
    
    for batch in big_train_loader: # take entire train set
        with torch.no_grad():
            # Embed training set under model
            rep_tr, _ = model(val_aug(batch.x, batch.edge_index, batch.edge_attr), batch.batch.to(device))
            if torch.cuda.is_available():
                rep_tr = rep_tr.to("cpu")
            rep_tr = pd.DataFrame(rep_tr.numpy())
            rep_tr.join(xenonpy_tr_df)
            
    
     
           
            for val_batch in big_val_loader: #take entire val set
                # Embed validation set under model
                rep_val, _ = model(val_aug(val_batch.x, val_batch.edge_index, val_batch.edge_attr), val_batch.batch.to(device))
                if torch.cuda.is_available():
                    rep_val = rep_val.to("cpu")
                rep_val = pd.DataFrame(rep_val.numpy())
                rep_val.join(xenonpy_val_df)
              
            
                # For each task in QM9
                for tar_ind in range(batch.y.shape[1]):
                    # Fit a model on model representation of train set

                    #print(rep_tr.shape, batch.y[tar_ind].shap)
                   
                    # Test the model on model representation of val set
                    lm = LinearRegression().fit(rep_tr, batch.y[:,tar_ind])
                    tar_yhat = lm.predict(rep_val)
                    mse_met = mean_squared_error(val_batch.y[:,tar_ind], tar_yhat).item()
                    r2_met = r2_score(val_batch.y[:,tar_ind], tar_yhat)
                    #print(qm9_index[tar_ind], mse_met, r2_met)
                    mse_scores[tar_ind, i_str] = mse_met
                    print("LM MSE for ", qm9_index[tar_ind], ":", mse_met)
                    
                    #print(rep_tr.shape, batch.y[tar_ind].shap)
                      # Test the model on model representation of val set
                    rf = RandomForestRegressor(n_estimators=rf_parameters['n_estimators'], max_depth=rf_parameters['max_depth'], warm_start=True).fit(rep_tr, batch.y[:,tar_ind])
                    #rf = rf_list[tar_ind]
                    tar_yhat = rf.predict(rep_val)
                    mse_met = mean_squared_error(val_batch.y[:,tar_ind], tar_yhat).item()
                    rf_mse_scores[tar_ind, i_str] = mse_met
                    print("RF MSE for ", qm9_index[tar_ind], ":", mse_met)
                    
                    lgb_train = lgb.Dataset(rep_tr, batch.y[:,tar_ind])
                    lgb_eval = lgb.Dataset(rep_val, val_batch.y[:,tar_ind], reference=lgb_train)
                    gbm = lgb.train(lgbm_parameters['params'],
                                    lgb_train,
                                    num_boost_round=lgbm_parameters['num_boost_round'],
                                    valid_sets=lgb_eval,
                                    callbacks=lgbm_parameters['callbacks'])
                    lgb_yhat = gbm.predict(rep_val, num_iteration=gbm.best_iteration)
                    lgb_score = round(mean_squared_error(val_batch.y[:,tar_ind], lgb_yhat), 2)
                    lgb_mse_scores[tar_ind, i_str] = lgb_score
                    print("LGBM MSE for ", qm9_index[tar_ind], ":", lgb_score)
                
                    x_val = pd.DataFrame(rep_val.numpy())
                    y_tr = pd.DataFrame(batch.y[:,tar_ind]).astype("float")
                    means_vector = y_tr.mean(axis = 0)
                    rep_means_vectors = means_vector.repeat(x_val.shape[0]) #create a vector where each entry is the mean
                    baseline = round(mean_squared_error(val_batch.y[:,tar_ind], rep_means_vectors), 2)
                    baseline_mse_scores[tar_ind, i_str] = baseline
                    print("LGBM MSE for ", qm9_index[tar_ind], ":", baseline)

    
plt.legend()
plt.show()




# In[ ]:


results_dict = {'LM_results':mse_scores, 'RF_results':rf_mse_scores, 'LGB_results':lgb_mse_scores, 'Basic_model':baseline_mse_scores}



# In[ ]:


# I would also add a column for "naive estimator"
# This could be a simple regressor, or a mean estimator (like Eric)

for i, row in enumerate(mse_scores):
    name = qm9_index[i]
    
    plt.bar(x = range(len(row)), height = row)
    plt.xticks(range(len(row)), ['NOT '+x for x in aug_strs], rotation = -30)
    plt.xlabel('Augmentation')
    plt.ylabel('Validation MSE Score')
    plt.title(name)
    
    plt.savefig(f'imgs/3-4_{qm9_index[i]}.png', bbox_inches = 'tight')
    plt.savefig(f'imgs/3-4_{qm9_index[i]}.pdf', bbox_inches = 'tight')
    plt.show()
    



# In[ ]:





# In[ ]:


# Further step would be summarizing the average ranks of the models to summarize 19 QM's into 1 rank
import scipy.stats as ss
# For each QM9_index, find the min of the row -> this index is the best augmentation
print('PLEASE NOTE THESE ARE RANKS INTEGERS NOT INDEXES:')

ranks = []
for i, row in enumerate(mse_scores):
    elem = torch.argmin(row).item()
    
    # For each element of the qm9 row, 
    rank = ss.rankdata(row)
    
    ranks.append(torch.FloatTensor(rank))
    
ranks = torch.stack(ranks)
print(ranks.shape)
meanranks= torch.mean(ranks, dim = 0)
print(meanranks)

plt.bar(range(4), meanranks)
plt.xticks(range(4), ['NOT '+x for x in aug_strs], rotation = -30)
plt.ylabel('Average Rank')
plt.xlabel('Augmentation')
plt.title('Average Augmentation Rank Across Transfer Tasks')
plt.savefig(f'imgs/3-4_{qm9_index[i]}.png', bbox_inches = 'tight')
plt.savefig(f'imgs/3-4_{qm9_index[i]}.pdf', bbox_inches = 'tight')
plt.show()


# In[ ]:





# In[ ]:


# record end time
t_1 = timeit.default_timer()
 
# calculate elapsed time
elapsed_time = round((t_1 - t_0) , 1)
print(f"Elapsed time: {elapsed_time} seconds")
elapsed_time_minutes = round((elapsed_time/60), 2)
print(f"Elapsed time: {elapsed_time_minutes} minutes")
elapsed_time_hours = round((elapsed_time/3600), 2)
print(f"Elapsed time: {elapsed_time_hours} hours")

other_info = {'dataset':dataset, 'hours':elapsed_time_hours, 'minutes':elapsed_time_minutes, 'seconds':elapsed_time}


# In[ ]:





# In[ ]:





# In[ ]:


run = True
if run == True:
    print("Saving results...")
    #save experimental results
    current_time = datetime.now()
    dt_string = current_time.strftime("%Y-%m-%d_%H_%M")
    directory = dt_string
    parent_dir = '/home/ewvertina/Molecular_modelling/Experiment_Results/'
    path = os.path.join(parent_dir, directory)
    os.mkdir(path)
    path_state_dict = path + '/state_dict.txt'
    path_results_dict = path + '/results_dict.txt'
    path_fit_params_dict = path + '/fit_params_dict.txt'
    path_runtime = path + '/runtime.txt'
    path_parameters = path + '/parameters_used.txt'
    path_fig = path + '/train_test_loss.png'
    
    #save NN model as a torch dictionary
    #torch.save(model.state_dict(), path_state_dict)
    file = open(path_state_dict, 'w')
    file.write(str(model.state_dict()))
    file.close()
    
    #torch.save(results_dict, path_results_dict)
    file = open(path_results_dict, 'w')
    file.write(str(results_dict))
    file.close()
    
    #torch.save(fit_params_dict, path_fit_params_dict)
    file = open(path_fit_params_dict, 'w')
    file.write(str(fit_params_dict))
    file.close()
    
    #torch.save(other_info, path_runtime) #save which dataset, runtime
    file = open(path_runtime, 'w')
    file.write(str(other_info))
    file.close()
    
    #torch.save(parameters_used, path_parameters) #saves all parameters used
    file = open(path_parameters, 'w')
    file.write(str(parameters_used))
    file.close()
    
    train_test_plot.savefig(path_fig, format='png')
    #plt.savefig(path_fig, format='png') #save train-val loss figure
    print("Saved!")


# In[ ]:





# In[ ]:





# In[ ]:


print("Hello, world!")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




