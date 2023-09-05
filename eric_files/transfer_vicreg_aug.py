#!/usr/bin/env python
# coding: utf-8

# In[ ]:


print("Code is running!")
import torch
import torch.nn as nn

import torch_geometric
from torch_geometric.nn import GCNConv, SoftmaxAggregation
from torch_geometric.datasets import QM9
import torch_geometric.nn as gnn
import torch.nn.functional as F

import GCL.augmentors as A
#import edge_removing as A_alternate

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression, LinearRegression
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


import numpy as np
from matplotlib import pyplot as plt

import timeit
import os
from datetime import datetime

import argparse
import math
import random
import pandas as pd
import csv
from csv import DictWriter
import os.path

import pandas as pd
from rdkit.Chem import PeriodicTable
from rdkit import Chem
from xenonpy.datatools import preset
from xenonpy.descriptor import Compositions
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from xenonpy.datatools import preset

import copy
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[ ]:


#record start time
t_0 = timeit.default_timer()
scratch_work = False


# In[ ]:


dataset = "QM9"


# In[ ]:


if scratch_work == False: #if running real experiment
    parser = argparse.ArgumentParser(description='Neural message passing')
    parser.add_argument('--augmentations', type=int, default=-1, metavar='augmentations',
                    help='Indices of augmentations to include')
    args = parser.parse_args()


# In[ ]:





# In[ ]:


#parser.add_argument('--batch-size', type=int, default=100, metavar='batch_size',
 #                   help='Input batch size for training (default: 20)')


# In[ ]:


#parser.add_argument('--epochs', type=int, default=50, metavar='epochs',
#                    help='Number of epochs to train (default: 360)')


# In[ ]:


#parser.add_argument('--lr', type=float, default=1e-4, metavar='lr',
#                    help='Initial learning rate [1e-5, 5e-4] (default: 1e-4)')


# In[ ]:


#parser.add_argument('--lr-decay', type=float, default=0.6, metavar='lr_decay',
  #                  help='Learning rate decay factor [.01, 1] (default: 0.6)')


# In[ ]:


#parser.add_argument('--num-filters', type=int, default=64, metavar='num_filters',
 #                   help='Number of filters, default 64')


# In[ ]:


#parser.add_argument('--momentum', type=float, default=0.9, metavar='momentum',
 #                   help='SGD momentum (default: 0.9)')


# In[ ]:





# In[ ]:


target_features_list = ['Dipole_moment', 
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
                  'Rotational_constant_C'
                 ]


# In[ ]:


target_features_dict = {0: 'Dipole_moment',
1: 'Isotropic_polarizability',
2: 'Highest_occupied_molecular_orbital_energy',
3: 'Lowest_unoccupied_molecular_orbital_energy',
4: 'Gap_between_previous_2',
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


#list of input features in QM9 dataset
x_index = {0: 'H_atom?',
1: 'C_atom?',
2: 'N_atom?',
3: 'O_atom?',
4: 'F_atom?',
5: 'atomic_number',
6: 'aromatic',
7: 'sp1',
8: 'sp2',
9: 'sp3',
10: 'num_hs'}

x_index_list = ['H_atom?', 
                'C_atom?', 
                'N_atom?', 
                'O_atom?', 
                'F_atom?', 
                'atomic_number', 
                'aromatic', 
                'sp1',
                'sp2',
                'sp3',
                'num_hs']


# In[ ]:


if scratch_work == False: #if running real experiment
    print("Augmentations input: ", args.augmentations)


# In[ ]:


parameters = {}
run_downstream_models = False #only run these for the best architecture model (don't forget to get rid of downstream NN layers!)
run_xenonpy = False #run these to add additional atomic properties to each node, based on the 58 additional input features from XenonPy
if run_xenonpy == True:
    XPy_df = preset.dataset_elements_completed #XenonPy basic atomic feature information
    


# In[ ]:





# In[ ]:


#augmentation 0 and 5 do not work when used together
# Augmentation selection
augs = [A.RWSampling(num_seeds=1000, walk_length=10),
        ##A.EdgeAttrMasking(pf=0.1),
        ##A.MarkovDiffusion(),
        A.NodeDropping(pn=0.1),
        A.NodeShuffling(),
        ##A.EdgeAdding(pe=0.1),
        A.FeatureMasking(pf=0.1),
        A.FeatureDropout(pf=0.1),
        A.EdgeRemoving(pe=0.1)
       ]

#args.augmentations

temp_list = []

if scratch_work == True: #if not running real experiment
    aug_list = str(2340)
else:
    aug_list = str(args.augmentations)

for index in aug_list:
    temp_list.append(augs[int(index)])

augs = temp_list
print("Updated augs: ", augs)



if scratch_work == True: #if not running real experiment
    augmentation = A.RandomChoice(augs, num_choices = 4)
else:
    augmentation = A.RandomChoice(augs, num_choices = len(str(args.augmentations)))

print("augmentation: ", augmentation)


val_aug = A.RandomChoice([], num_choices = 0)
print("val_aug: ", val_aug)

parameters['augmentation'] = augmentation
aug = parameters['augmentation']



# In[ ]:





# In[ ]:


print(parameters['augmentation'])


# In[ ]:


# Hyperparameters

if scratch_work == True: #if not running real experiment
    parameters['n_epochs'] = 1
else:
    parameters['n_epochs'] = 50

parameters['learning_rate'] = 0.0001
parameters['batch_size'] = 256
parameters['model_size'] = 16
parameters['learning_rate_decay'] = 0.01
parameters['momentum'] = 0.9



# Supervised criterion
metrics = [mean_squared_error, mean_absolute_error, r2_score]
metrics_used = ['mse', 'mae', 'r2']


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

xgboost_parameters = {'default': 'used default for all parameters'}


# In[ ]:





# In[ ]:


whole_dataset = QM9(root = 'data/')

idx = []
for i in range(130831):
    if i != 474 and i != 14240:
        idx += [i]
whole_dataset = whole_dataset.index_select(idx= idx)

# outlier removal

n = whole_dataset.len()
tr_n = 0.9  # Number of QM9 to use as training data
val_n = 0.05
test_n = 0.04
final_test_n = 0.01

print("Training set proportion: ", tr_n)
print("Validation set proportion: ", val_n)
print("Training set proportion: ", test_n)

all_inds = range(n-2)
tr_inds, val_inds = train_test_split(all_inds, train_size = tr_n, random_state = 24)
val_test_inds = range(n - math.floor(tr_n*n))
val_inds, test_inds = train_test_split(val_test_inds, train_size = val_n, random_state = 24)
test_final_test_inds = range(n - math.floor(tr_n*n) - math.floor(val_n*n))
test_inds, final_test_inds = train_test_split(test_final_test_inds, train_size = test_n, random_state = 24)

train_set = torch.utils.data.Subset(whole_dataset, tr_inds)
val_set = torch.utils.data.Subset(whole_dataset, val_inds)
test_set = torch.utils.data.Subset(whole_dataset, test_inds)
final_test_set = torch.utils.data.Subset(whole_dataset, final_test_inds)

train_loader = torch_geometric.loader.DataLoader(train_set, batch_size = parameters['batch_size'],
                                                shuffle = True, num_workers = 1)

big_train_loader = torch_geometric.loader.DataLoader(train_set, batch_size = int(1e9),
                                                shuffle = True, num_workers = 1)

val_loader = torch_geometric.loader.DataLoader(val_set, batch_size=parameters['batch_size'], 
                                            shuffle=True, num_workers=1)

big_val_loader = val_loader = torch_geometric.loader.DataLoader(val_set, batch_size=int(1e9), 
                                            shuffle=True, num_workers=1)

test_loader = torch_geometric.loader.DataLoader(test_set, batch_size=int(1e9), 
                                            shuffle=True, num_workers=1)



# In[ ]:





# In[ ]:





# In[ ]:


#get mean and standard deviation of training data
def get_tr_mean_std(entire_train_loader, target_features_list):
    for batch in entire_train_loader:
        df = pd.DataFrame(batch.y.float(), columns = target_features_list)
        mu = df.mean().to_frame().T
        std = df.std().to_frame().T

    return mu, std


# In[ ]:





# In[ ]:


class GCN(torch.nn.Module): #probably need to alter architecture
    def __init__(self, model_size, run_xenonpy):
        super().__init__()
        
        self.rep_dim = model_size
        self.emb_dim = model_size * 2
        self.run_xenonpy = run_xenonpy
        
        # Data under graph
        if run_xenonpy == False:
            self.conv1 = GCNConv(whole_dataset.num_node_features, self.rep_dim // 2)
        else:
            self.conv1 = GCNConv(whole_dataset.num_node_features + len(XPy_df.columns) - 1, self.rep_dim // 2)
        self.conv1.aggr = SoftmaxAggregation(learn=True)
        self.bn1 = nn.BatchNorm1d(self.rep_dim // 2)
        self.a1 = nn.LeakyReLU(0.02)
        
        self.conv2 = GCNConv(self.rep_dim // 2, self.rep_dim) # To Rep Space
        self.conv2.aggr = SoftmaxAggregation(learn=True)
        self.bn2 = nn.BatchNorm1d(self.rep_dim)
        
        
        # Projection to representation
        self.mpool1 = gnn.global_mean_pool
        #self.fc1 = nn.Linear(self.rep_dim, self.rep_dim)
        
        # Graph 2
        self.conv3 = GCNConv(self.rep_dim, self.rep_dim * 2) # To Emb Space
        self.bn3 = nn.BatchNorm1d(self.rep_dim * 2)
        
        #VicReg needs to be bigger and deeper
            #more graph convolutional layers
            #more filters
                #search for "state of the art graph convolutional network" and see if we can improve by using their method
                #Could even try other pooling methods
        #Instead of a linear layer at end (or LM), should do 2-layer NN
        
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


# In[ ]:


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


def train(parameters, run_xenonpy):
    print("Training")
    device = 'cuda'

    model = GCN(parameters['model_size'], run_xenonpy).to(device)
    n_epochs = parameters['n_epochs']
    aug = parameters['augmentation'] #aug is a RandomChoice object from GCL.augmentors
    
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters['learning_rate'], weight_decay=parameters['learning_rate_decay'])

    tr_losses = []
    val_losses = []

    for epoch in range(0,n_epochs+1):
        print("Training epoch ", epoch)
        epoch_losses = []
        for batch in train_loader:
            optimizer.zero_grad()

            batch_inds = batch.batch.to(device)

            # batch of graphs has edge attribs, node attribs - (n_nodes, n_features+1) -> concat (n_nodes, attrib1)
            if run_xenonpy == True:
                batch.x = node_transform_XenonPy(batch.x.cpu(), XPy_df, x_index)
            
            batch.x = batch.x.float()#.to(device)
            
            # Barlow - get 2 random views of batch
            b1 = aug(batch.x, batch.edge_index, batch.edge_attr)
            b2 = aug(batch.x, batch.edge_index, batch.edge_attr)
                      
            # Embed each batch (ignoring representations)
            r1, e1 = model(b1, batch_inds)
            r2, e2 = model(b2, batch_inds)
                        
            loss = VicRegLoss(e1, e2)
            loss.backward()
            optimizer.step()
           
            epoch_losses.append(loss.data.item())
        
        
        #print('epoch train loss', sum(epoch_losses) / len(epoch_losses))
        tr_losses.append(sum(epoch_losses) / len(epoch_losses))
        print("Epoch ", epoch, "training loss: ", sum(epoch_losses) / len(epoch_losses))
        
        # VicReg Validation Loss
        if True:
            val_loss = []
            for batch in val_loader:
                with torch.no_grad():
                    # VicReg validation loss
                    #should these be val_aug(*), instead of aug(*)?
                    if run_xenonpy == True:
                        batch.x = node_transform_XenonPy(batch.x.cpu(), XPy_df, x_index)
                    b1 = aug(batch.x, batch.edge_index, batch.edge_attr)
                    b2 = aug(batch.x, batch.edge_index, batch.edge_attr)
                    r1, e1 = model(b1, batch.batch.to(device))
                    r2, e2 = model(b2, batch.batch.to(device))

                    val_loss.append(VicRegLoss(e1, e2).item())

            val_losses.append(torch.mean(torch.FloatTensor(val_loss)))
        validation_float_loss = float(torch.mean(torch.FloatTensor(val_loss)))
        print("Epoch ", epoch, "validation loss: ", validation_float_loss)
    return model, tr_losses, val_losses

def test(model, big_train_loader, val_loader, parameters):
    print('entering test, ')
    # Downstream supervised loss      
    scores = []
    for batch in big_train_loader: # take entire train set
        with torch.no_grad():
            # Embed training set under model
            #should this be aug(*), instead of val_aug(*)?
            rep_tr, _ = model(val_aug(batch.x, batch.edge_index, batch.edge_attr), batch.batch.to(device))

            for i, val_batch in enumerate(big_val_loader):
                # Embed validation set under model
                rep_val, _ = model(val_aug(val_batch.x, val_batch.edge_index, val_batch.edge_attr), val_batch.batch.to(device))

                # For each task in QM9
                for tar_ind in range(batch.y.shape[1]):
                    # Fit a model on model representation of train set
                    lm = LinearRegression().fit(rep_tr.cpu(), batch.y[:,tar_ind])
                    # Test the model on model repersentation of val set
                    tar_yhat = lm.predict(rep_val.cpu())
                    mse_met = mean_squared_error(val_batch.y[:,tar_ind], tar_yhat).item()
                    r2_met = r2_score(val_batch.y[:,tar_ind], tar_yhat)
                    scores.append(mse_met)
                    print("Linear model loss for ", target_features_dict[tar_ind], ": ", mse_met)
                    print("R2 score for ", target_features_dict[tar_ind], ": ", r2_met)
                if i==0:
                    break # Only want first batch, please
                    
    return scores

def transfer(model, val_loader, parameters, run_xenonpy):
    # Transfer a model trained under the supervised paradigm    
    # Need to get training set embeddings:
    train_batch = next(iter(big_train_loader))
    
    if run_xenonpy == True:
        train_batch.x = node_transform_XenonPy(train_batch.x.float().to(device), XPy_df, x_index)
    
    
    with torch.no_grad():
        tr_emb, _ = model([train_batch.x.float().to(device), train_batch.edge_index, train_batch.edge_attr], train_batch.batch.to(device))
        tr_emb = tr_emb.cpu()
    
    val_batch = next(iter(val_loader))
    
    
    batch_inds = val_batch.batch.to(device)
    val_batch.x = val_batch.x.float().to(device)
    
    if run_xenonpy == True:
        val_batch.x = node_transform_XenonPy(val_batch.x, XPy_df, x_index)
    
    
    with torch.no_grad():
        val_emb, _ = model([val_batch.x, val_batch.edge_index, val_batch.edge_attr], batch_inds)
        val_emb = val_emb.cpu()

 
    scoremat = torch.zeros((len(target_features_dict.keys()), len(metrics)))
    for task in target_features_dict.keys():
        linear_classifier = LinearRegression().fit(tr_emb, train_batch.y[:,task])
        yhat = linear_classifier.predict(val_emb)
        for meti, metric in enumerate(metrics):
            met = metric(yhat, val_batch.y[:,task])
            scoremat[task, meti] = met.astype(np.float64)
             

    print('Returning transfer scores', scoremat.shape)
    return scoremat




# In[ ]:


def baseline_models(y_train, y_val):
    #creates models that simply always output the mean of the training set
    #returns a list containing the performance by specified criteria of a model that always outputs the mean of the training set for each property
    
    #initialize list for this data row
    model_results = ['Means_baseline', 'N/A', 'Baseline'] #these are the primary keys for this row/example, corresponding to Model type, Augmentations used, and Trial #, respectively

    for feature in y_train: #iterate over target features
        means_vector = y_train[feature].mean(axis = 0)      
        rep_means_vectors = means_vector.repeat(y_val[feature].shape[0])
        
        for criterion in metrics: #for each criteria, output its score given the baseline model for this feature
            #e.g., criterion list could contain [MSE, MAE]
            model_results += [criterion(y_val[feature], rep_means_vectors)]
               
    
    return model_results
        
        


# In[ ]:


def rf_models(x_train, x_val, y_train, y_val, params, aug_set, trial):
    #trains, predicts, and evaluates one Random Forest model for each of the target features based on specified criteria
    #returns a list containing the RF performance based on the criteria specified for all of the target properties
    #this list will be input as one row (example) into a dataframe containing all results
    
    #initialize list for this data row
    model_results = ['RF', aug_set, trial]  #these are the primary keys for this row/example, corresponding to Model type, Augmentations used, and Trial #, respectively
    
    for feature in y_train: #train an RF model for each target feature
        rf = RandomForestRegressor(n_estimators = 100, max_depth = 10)
        rf.fit(x_train, y_train[feature])
        rf_yhat = rf.predict(x_val.values)
        
        for criterion in metrics: #for each criteria, output its score given the RF model for this feature
            #e.g., criterion list could contain [MSE, MAE]
            model_results += [criterion(y_val[feature].values, rf_yhat)]  
     
   
    return model_results


# In[ ]:


def lgbm_models(x_train, x_val, y_train, y_val, params, aug_set, trial):
    #trains, predicts, and evaluates one LGBM model for each of the target features based on specified criteria
    #returns a list containing one row of results data that will go into a dataframe containing all results
    #this list contains the LGBM performance based on all criteria specified for all of the target properties
    
    #initialize list for this data row
    model_results = ['LGBM', aug_set, trial] #these are the primary keys for this row/example, corresponding to Model type, Augmentations used, and Trial #, respectively
    

    
    for feature in y_train: #train an LGBM model for each target feature
        lgb_train = lgb.Dataset(x_train.values, y_train[feature].values, params={'verbose': -1})
        lgb_eval = lgb.Dataset(x_val.values, y_val[feature].values, reference=lgb_train, params={'verbose': -1})


        gbm = lgb.train(params['params'],
                        lgb_train,
                        num_boost_round=params['num_boost_round'],
                        valid_sets=lgb_eval,
                        callbacks=params['callbacks'])
        lgb_yhat = gbm.predict(x_val.values, num_iteration=gbm.best_iteration)
        
        for criterion in metrics: #for each criteria, output its score given the LGBM model for this feature
            #e.g., criterion list could contain [MSE, MAE]
            model_results += [criterion(y_val[feature].values, lgb_yhat)]            
    
   
    return model_results


# In[ ]:


def xgboost_models(x_train, x_val, y_train, y_val, params, aug_set, trial):
    #trains, predicts, and evaluates one XGBoost model for each of the target features based on specified criteria
    #returns a list containing one row of results data that will go into a dataframe containing all results
    #this list contains the XGBoost performance based on all criteria specified for all of the target properties
    
    #initialize list for this data row
    model_results = ['XGBoost', aug_set, trial] #these are the primary keys for this row/example, corresponding to Model type, Augmentations used, and Trial #, respectively
    
    for feature in y_train: #train an XGBoost model for each target feature
        xgb_model = xgb.XGBRegressor(random_state=42)
        xgb_model.fit(x_train, y_train[feature])
        y_pred = xgb_model.predict(x_val)
        
        for criterion in metrics: #for each criteria, output its score given the RF model for this feature
            #e.g., criterion list could contain [MSE, MAE]
            model_results += [criterion(y_val[feature], y_pred)]  
    
  
    return model_results


# In[ ]:


def normalize_target_df(target_df, mean, std):
    target_df = target_df.rename(columns = target_features_dict)
    df_normalized = target_df.copy(deep=True)

    for feature in target_df:
        if feature == 'Augmentations_used' or feature == 'Trial':
            pass
        else:
            df_normalized[feature] = df_normalized[feature].sub(float(mean[feature]))
            df_normalized[feature] = df_normalized[feature].div(float(std[feature]))
            
    return df_normalized


# In[ ]:


def save_or_append_df(df, filepath):
    if not os.path.isfile(filepath): #see if the .csv file to be used already exists. If it does, then append to existing file. If not, create a new file and append to that
        #csv file does not yet exist. Create a csv file
        print("Creating new mae_results.csv file")
        with open(filepath, 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvfile.close()
        print("Appending results to file")
        df.to_csv(filepath, mode='a', index=False, header=True) #creates headers
    else:
        print("Appending results to file")
        df.to_csv(filepath, mode='a', index=False, header=False)
    
    print("File saved to ", filepath)
    return filepath



# In[ ]:


def get_fit_params(fit_dict):
    fit_params = {}
    for (_, target_feature) in enumerate(fit_dict):
        fit_params[target_feature] = fit_dict[target_feature].get_params
    
    return fit_params


# In[ ]:


def node_transform_XenonPy(x_df, XPy_df, x_index):
    
    #feature_names = list(XPy_df.columns)
    x_df = pd.DataFrame(x_df.cpu())
    xenonpy_df = x_df.rename(x_index, axis='columns')
    xenonpy_df = xenonpy_df.merge(XPy_df, how='left', on='atomic_number')
    xenonpy_tensor = torch.tensor(xenonpy_df.values)
    
    return xenonpy_tensor


# In[ ]:





# In[ ]:





# In[ ]:


path = '/home/ewvertina/Molecular_modelling/Experiment_Results/2023-08-24/'


# In[ ]:


tr_mean, tr_std = get_tr_mean_std(big_train_loader, target_features_list)


# In[ ]:


val_mean, val_std = get_tr_mean_std(big_val_loader, target_features_list)


# In[ ]:


results_columns = ['Model', 'Augmentations_used', 'Trial']


# In[ ]:


#Make new column names for target features to incorporate criteria used
    #add string name of criterion to each of the target properties; do this for all target features
        #e.g. If have 19 target properties, and 3 criteria, say MSE, MAE, and R2, then there will now be 57 columns
            #the first four columns will be: mse_targetfeature1, mae_targetfeature1, r2_targetfeature1, mse_targetfeature2, etc.
for j in range(len(target_features_list)):
    for i in range(len(metrics_used)):
        results_columns.append(str(metrics_used[i]) + '_' + target_features_list[j])
        


# In[ ]:


results_list = []


# In[ ]:


if run_downstream_models == True:
    for tr_batch in big_train_loader:
        y_tr = pd.DataFrame(tr_batch.y).astype("float")

    y_tr = y_tr.rename(columns = target_features_dict)
    y_tr = normalize_target_df(y_tr, tr_mean, tr_std) #standard normalize y_tr
    
    for val_batch in big_val_loader: #get entire test set
        y_val = pd.DataFrame(val_batch.y).astype("float")
    
    y_val = y_val.rename(columns = target_features_dict)
    y_val = normalize_target_df(y_val, val_mean, val_std)  #standard normalize y_val
    


# In[ ]:





# In[ ]:


if run_xenonpy == True:
    x_tr_xenonpy = node_transform_XenonPy(tr_batch.x, XPy_df, x_index)
    x_val_xenonpy = node_transform_XenonPy(val_batch.x, XPy_df, x_index)

    


# In[ ]:





# In[ ]:


if scratch_work == True: #if not running real experiment
    n_trials = 1
else:
    n_trials = 10
for i in range(1, n_trials + 1):
    print('Trial ', i)
    
    model, tr_loss, val_loss = train(parameters, run_xenonpy)
    
    print(tr_loss, val_loss)
    plt.plot(tr_loss, label = 'tr')
    plt.plot(val_loss, label = 'val')
    plt.legend(loc = 'best')
    plt.show()

    scores = transfer(model, val_loader, parameters, run_xenonpy)
    scores_df = pd.DataFrame(scores.numpy().T, columns = target_features_list)
    scores_list = scores_df.to_numpy().flatten(order = 'F')
    scores_list = [element for element in scores_list]
    results_list.append(['GCL_VicReg', aug_list, i] + scores_list)

        
    if i == (n_trials - 1):
        plt.savefig(path + 'loss_' + aug_list + '.png', format = 'png')


   
    #train and evaluate downstream models  
    
    if run_downstream_models == True:
        print("Fitting models!")

        
        #get embeddings, train downstream models on embeddings
        with torch.no_grad():
            # Embed training set under model
            #when doing supervised learning, do empty augmentation
            #when doing unsupervised learning (training representation model), do (nonempty) augmentations
            if run_xenonpy == False:
                rep, _ = model(val_aug(tr_batch.x, tr_batch.edge_index, tr_batch.edge_attr), tr_batch.batch.to(device))
            else:
                rep, _ = model(val_aug(x_tr_xenonpy, tr_batch.edge_index, tr_batch.edge_attr), tr_batch.batch.to(device))

            if torch.cuda.is_available():
                rep = rep.to("cpu")
        x_tr = pd.DataFrame(rep.numpy())


        with torch.no_grad():    
            # Embed validation set under model
            #when doing supervised learning, do empty augmentation
            #when doing unsupervised learning (training representation model), do (nonempty) augmentations

            if run_xenonpy == False:
                rep, _ = model(val_aug(val_batch.x, val_batch.edge_index, val_batch.edge_attr), val_batch.batch.to(device))
            else:
                rep, _ = model(val_aug(x_val_xenonpy, val_batch.edge_index, val_batch.edge_attr), val_batch.batch.to(device))

            if torch.cuda.is_available():
                rep = rep.to("cpu")
        x_val = pd.DataFrame(rep.numpy())
        
        
        
        #train and evaluate lgbm models for every target feature
        lgbm_results = lgbm_models(x_tr, x_val, y_tr, y_val, lgbm_parameters, aug_list, i)
        results_list.append(lgbm_results)

        #train and evaluate xgboost models for every target feature
        xgboost_results = xgboost_models(x_tr, x_val, y_tr, y_val, xgboost_parameters, aug_list, i)
        results_list.append(xgboost_results)

        #train and evaluate rf models for every target feature  
        rf_results = rf_models(x_tr, x_val, y_tr, y_val, rf_parameters, aug_list, i)
        results_list.append(rf_results)


    print("Done predicting validation set for trial ", i, "!")

    


# In[ ]:





# In[ ]:





# In[ ]:


#make sure to change the filepath name with every experiment!
results_filepath = path + 'results.csv'


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


# In[ ]:





# In[ ]:


other_info = {'dataset':dataset, 'hours':elapsed_time_hours, 'minutes':elapsed_time_minutes, 'seconds':elapsed_time}


# In[ ]:


#get and save summary statistics of dataset
summary_statistics_filepath = path + 'summary_statistics_' + dataset + '_dataset.csv'

if not os.path.isfile(summary_statistics_filepath): #see if the .csv file to be used already exists. If it does, then append to existing file. If not, create a new file and append to that
    #csv file does not yet exist. Create a csv file
    #get summary statistics of dataset

    for tr_batch in big_train_loader:
        y_tr = pd.DataFrame(tr_batch.y).astype("float")

    y_tr = y_tr.rename(columns = target_features_dict)   
    
    summary_statistics_df = round(y_tr.describe(), 2)
    print("Summary statistics for dataset: ", summary_statistics_df)
    print("Creating new summary statistics .csv file")
    save_or_append_df(summary_statistics_df, summary_statistics_filepath)
    print("Summary statistics .csv file created")
    
    
    #put baseline prediction model here
    #put RF baseline model here
    #put LM baseline model here
    #put LGBM baseline model here
    #put XGBoost baseline model here
    
     
    #print('Training baselines')
    
    #baseline_results = baseline_models(y_tr, y_val)
    #results_list.append(baseline_results)
    
    #train and evaluate lgbm models for every target feature
    #lgbm_results = lgbm_models(x_tr, x_val, y_tr, y_val, lgbm_parameters, 'N/A', 'Baseline')
    #results_list.append(lgbm_results)

    #train and evaluate xgboost models for every target feature
    #xgboost_results = xgboost_models(x_tr, x_val, y_tr, y_val, xgboost_parameters, 'N/A', 'Baseline')
    #results_list.append(xgboost_results)

    #train and evaluate rf models for every target feature  
    #rf_results = rf_models(x_tr, x_val, y_tr, y_val, rf_parameters, 'N/A', 'Baseline')
    #results_list.append(rf_results)
    
else:
    print("Summary statistics .csv file already saved")
    


# In[ ]:





# In[ ]:


results_df = pd.DataFrame(results_list, columns = results_columns)


# In[ ]:


print(results_df)


# In[ ]:





# In[ ]:





# In[ ]:


if scratch_work == True : #if not running real experiment
    run = False
else:
    run = True

if run == True:
    print("Saving results...")
    #save experimental results
    save_or_append_df(results_df, results_filepath)

    path_state_dict = path + '/state_dict_' + aug_list + '.txt'
    path_fit_params_dict = path + '/fit_params_dict_' + aug_list + '.txt'
    path_runtime = path + '/runtime_' + aug_list + '.txt'
    path_parameters = path + '/parameters_used_' + aug_list + '.txt'
    path_fig = path + '/train_test_loss_' + aug_list + '.png'
    
    #save NN model as a torch dictionary
    #torch.save(model.state_dict(), path_state_dict)
    file = open(path_state_dict, 'w')
    file.write(str(model.state_dict()))
    file.close()
    
    ##torch.save(fit_params_dict, path_fit_params_dict)
    #file = open(path_fit_params_dict, 'w')
    #file.write(str(fit_params_dict))
    #file.close()
    
    #torch.save(other_info, path_runtime) #save which dataset, runtime
    file = open(path_runtime, 'w')
    file.write(str(other_info))
    file.close()
    
    #torch.save(parameters_used, path_parameters) #saves all parameters used
    file = open(path_parameters, 'w')
    file.write(str(parameters))
    file.close()
    
    print("Saved!")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




