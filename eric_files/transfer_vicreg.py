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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

import timeit
import os
from datetime import datetime

import argparse

import copy
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[ ]:


#record start time
t_0 = timeit.default_timer()


# In[ ]:





# In[ ]:


parser = argparse.ArgumentParser(description='Neural message passing')


# In[ ]:


parser.add_argument('--batch-size', type=int, default=100, metavar='batch_size',
                    help='Input batch size for training (default: 20)')


# In[ ]:


parser.add_argument('--epochs', type=int, default=360, metavar='epochs',
                    help='Number of epochs to train (default: 360)')


# In[ ]:


parser.add_argument('--lr', type=float, default=1e-4, metavar='lr',
                    help='Initial learning rate [1e-5, 5e-4] (default: 1e-4)')


# In[ ]:


parser.add_argument('--lr-decay', type=float, default=0.6, metavar='lr_decay',
                    help='Learning rate decay factor [.01, 1] (default: 0.6)')


# In[ ]:


parser.add_argument('--num-filters', type=int, default=64, metavar='num_filters',
                    help='Number of filters, default 64')


# In[ ]:


parser.add_argument('--momentum', type=float, default=0.9, metavar='momentum',
                    help='SGD momentum (default: 0.9)')


# In[ ]:


args = parser.parse_args()


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


parameters = {}

# Augmentation selection
augs = [#A.RWSampling(num_seeds=1000, walk_length=10),
        #A.EdgeAttrMasking(pf=0.1),
        #A.MarkovDiffusion(),
        A.NodeDropping(pn=0.1),
        A.NodeShuffling(),
        #A.EdgeAdding(pe=0.1),
        A.FeatureMasking(pf=0.1),
        A.FeatureDropout(pf=0.1),
        A.EdgeRemoving(pe=0.1)
]

augmentation = A.RandomChoice(augs, num_choices=2)

val_aug = A.RandomChoice([], num_choices = 0)

parameters['augmentation'] = augmentation

# Hyperparameters
parameters['n_epochs'] = args.epochs
parameters['learning_rate'] = args.lr
parameters['batch_size'] = args.batch_size
parameters['model_size'] = args.num_filters
parameters['learning_rate_decay'] = args.lr_decay
parameters['momentum'] = args.momentum

#parameters['n_epochs'] = 50
#parameters['learning_rate'] = 3e-4
#parameters['batch_size'] = 100
#parameters['model_size'] = 64
#parameters['learning_rate_decay'] = 0.6
#parameters['momentum'] = 0.9

# Supervised criterion
metrics = [mean_squared_error, mean_absolute_error, r2_score]



# In[ ]:


whole_dataset = QM9(root = 'data/')

idx = []
for i in range(130831):
    if i != 474 and i != 14240:
        idx += [i]
whole_dataset = whole_dataset.index_select(idx= idx)

# Eric outlier removal

n = whole_dataset.len()
tr_n = 0.5  # Number of QM9 to use as training data

all_inds = range(n-2)
tr_inds, val_inds = train_test_split(all_inds, train_size = tr_n)
train_set = torch.utils.data.Subset(whole_dataset, tr_inds)
val_set = torch.utils.data.Subset(whole_dataset, val_inds)

train_loader = torch_geometric.loader.DataLoader(train_set, batch_size = parameters['batch_size'],
                                                shuffle = True, num_workers = 2,)

big_train_loader = torch_geometric.loader.DataLoader(train_set, batch_size = int(1e9),
                                                shuffle = True, num_workers = 2,)

val_loader = torch_geometric.loader.DataLoader(val_set, batch_size=int(1e9), # I am using this to get a random subset of the val set
                                            shuffle=True, num_workers=2,)


# In[ ]:


class GCN(torch.nn.Module):
    def __init__(self, model_size):
        super().__init__()
        
        self.rep_dim = model_size
        self.emb_dim = model_size * 2
        
        # Data under graph
        self.conv1 = GCNConv(whole_dataset.num_node_features, self.rep_dim // 2)
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


def train(parameters):
    
    device = 'cuda'

    model = GCN(parameters['model_size']).to(device)
    n_epochs = parameters['n_epochs']
    aug = parameters['augmentation']
    
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters['learning_rate'], weight_decay=parameters['learning_rate_decay'])

    tr_losses = []
    val_losses = []

    for epoch in range(0,n_epochs+1):
        epoch_losses = []
        for batch in train_loader:
            #print('training batch')
            optimizer.zero_grad()

            batch_inds = batch.batch.to(device)

            # batch of graphs has edge attribs, node attribs - (n_nodes, n_features+1) -> concat (n_nodes, attrib1)
            batch.x = batch.x.float()#.to(device)
            
            #print('batch.x.shape', batch.x.shape)

            # Barlow - get 2 random views of batch
            b1 = aug(batch.x, batch.edge_index, batch.edge_attr)
            b2 = aug(batch.x, batch.edge_index, batch.edge_attr)

            # Embed each batch (ignoring representations)
            r1, e1 = model(b1, batch_inds)
            r2, e2 = model(b2, batch_inds)
            
            #print('calcing loss')

            loss = VicRegLoss(e1, e2)
            #print('backward loss')
            loss.backward()
            optimizer.step()
            #print('succesful backward')

            epoch_losses.append(loss.data.item())
        
        #print('epoch train loss', sum(epoch_losses) / len(epoch_losses))
        tr_losses.append(sum(epoch_losses) / len(epoch_losses))
        print("Epoch ", epoch, "training loss: ", sum(epoch_losses) / len(epoch_losses))
        
        # VicReg Validation Loss
        if True:
            val_loss = []
            for batch in val_loader:
                with torch.no_grad():
                    #print('calcuing validation')
                    # VicReg validation loss
                    b1 = aug(batch.x, batch.edge_index, batch.edge_attr)
                    b2 = aug(batch.x, batch.edge_index, batch.edge_attr)
                    r1, e1 = model(b1, batch.batch.to(device))
                    r2, e2 = model(b2, batch.batch.to(device))

                    val_loss.append(VicRegLoss(e1, e2).item())

            val_losses.append(torch.mean(torch.FloatTensor(val_loss)))
            #print('successful validation')
        print("Epoch ", epoch, "validation loss: ", torch.mean(torch.FloatTensor(val_loss)))
    return model, tr_losses, val_losses

def test(model, big_train_loader, val_loader, parameters):
    print('entering test, ')
    # Downstream supervised loss      
    scores = []
    for batch in big_train_loader: # take entire train set
        with torch.no_grad():
            # Embed training set under model
            rep_tr, _ = model(val_aug(batch.x, batch.edge_index, batch.edge_attr), batch.batch.to(device))

            for i, val_batch in enumerate(big_val_loader):
                #print('doing a batch')
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
                    print("Linear model loss for ", qm9_index[tar_ind], ": ", mse_met)
                    print("R2 score for ", qm9_index[tar_ind], ": ", r2_met)
                if i==0:
                    break # Only want first batch, please
                    
            #print('left the first test one')
                    
    return scores

def transfer(model, val_loader, parameters):
    # Transfer a model trained under the supervised paradigm    
    # Need to get training set embeddings:
    train_batch = next(iter(big_train_loader))
    with torch.no_grad():
        tr_emb, _ = model([train_batch.x.float().to(device), train_batch.edge_index, train_batch.edge_attr], train_batch.batch.to(device))
        #print('train embeddings', tr_emb.shape)
        tr_emb = tr_emb.cpu()
    
    val_batch = next(iter(val_loader))

    batch_inds = val_batch.batch.to(device)
    val_batch.x = val_batch.x.float()#.to(device)


    with torch.no_grad():
        val_emb, _ = model([val_batch.x, val_batch.edge_index, val_batch.edge_attr], batch_inds)
        #print('batch embeding:', val_emb.shape)
        val_emb = val_emb.cpu()

    scoremat = torch.zeros((len(qm9_index.keys()), len(metrics)))
    for task in qm9_index.keys():
        linear_classifier = LinearRegression().fit(tr_emb, train_batch.y[:,task])
        yhat = linear_classifier.predict(val_emb)
        for meti, metric in enumerate(metrics):
            met = metric(yhat, val_batch.y[:,task])
            scoremat[task, meti] = met.astype(np.float64)
            
    print('Returning transfer scores', scoremat.shape)
    return scoremat


# In[ ]:


#lr_str = 'lr' + str(parameters['learning_rate'])


# In[ ]:





# In[ ]:





# In[ ]:


n_trials = 3
for i in range(1,n_trials):
    model, tr_loss, val_loss = train(parameters)
    
    print(tr_loss, val_loss)
    plt.plot(tr_loss, label = 'tr')
    plt.plot(val_loss, label = 'val')
    plt.legend(loc = 'best')
    plt.show()
    plt.savefig('/home/ewvertina/Molecular_modelling/heatmap_results/lr' + str(parameters['learning_rate']) + '_epo' + str(parameters['n_epochs']) + '_bs' + str(parameters['batch_size']) + '_nfilt' + str(parameters['model_size']) + '_lrd' + str(parameters['learning_rate_decay']) + '_vicreg' + str(i) + '.png')
    scores = transfer(model, val_loader, parameters)
    torch.save(scores, '/home/ewvertina/Molecular_modelling/heatmap_results/lr' + str(parameters['learning_rate']) + '_epo' + str(parameters['n_epochs']) + '_bs' + str(parameters['batch_size']) + '_nfilt' + str(parameters['model_size']) + '_lrd' + str(parameters['learning_rate_decay']) + '_vicreg' + str(i) + '.pt')
    
    
    


# In[ ]:


#torch.save(scores, '/home/ewvertina/Molecular_modelling/heatmap_results/vicreg1.pt')
#torch.save(scores, '/home/ewvertina/Molecular_modelling/heatmap_results/lr' + str(parameters['learning_rate']) + '_epo' + str(parameters['n_epochs']) + '_bs' + str(parameters['batch_size']) + '_nfilt' + str(parameters['model_size']) + '_lrd' + str(parameters['learning_rate_decay']) + '_vicreg' + str(i) + '.pt')

    
    


# In[ ]:


#plt.plot(tr_loss, label = 'tr')
#plt.plot(val_loss, label = 'val')
#plt.legend(loc = 'best')
#plt.show()



# In[ ]:


print(scores)
print(scores.shape)


# In[ ]:


for met, row in zip(metrics, scores.T):
    #print(met, row)
    for i, item in enumerate(row):
        print(met, qm9_index[i], item)
        
#torch.save(scores, '/home/ewvertina/Molecular_modelling/Vicreg_score_demo/vicreg_score_demo_lr' + str(parameters['learning_rate']) + '_epo' + str(parameters['n_epochs']) + '_bs' + str(parameters['batch_size']) + '_nfilt' + str(parameters['model_size']) + '_lrd' + str(parameters['learning_rate_decay']) + '_vicreg' + str(i) + '.pt')



# In[ ]:





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





# In[ ]:




