# Barlow Utilities
import torch
import torch.nn as nn
import torch.nn.functional as F


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn import metrics
from sklearn.metrics import matthews_corrcoef, f1_score


import augutils
import barlow

import matplotlib.pyplot as plt
import random


device = 'cuda' if torch.cuda.is_available() else 'cpu'


from inspect import getmembers, isfunction
import augmentations
memlist = getmembers(augmentations, isfunction)
#augprob = 0.20 # maybe this augprob gets set to some function of length(memlist), where the expected value of augs is like 2 
augprob = 1.5/len(memlist) # EV is 1.5
print('Augmentation probability', augprob)

sec = 5

#criterion = BarlowTwinsLoss(device = device, lambda_param = 0.005)

def getdata(expdate, device = device, dtype = 'singles'):
    # Load the experiment and return one train-val split, doubles testing set, and singles+doubles binarization
    xtr, ytr, xte, yte = torch.load(f'data_std/{expdate}_{dtype}_package.pt')
    #print(xtr.shape, ytr.shape, xte.shape, yte.shape)
    #print(torch.unique(ytr.int(), dim = 1))
    xtr, xval, ytr, yval = train_test_split(xtr, ytr, test_size = 0.33, stratify=binarize(ytr)) # , stratify = ytr

    singles_binary = torch.FloatTensor([1 if x[0] > 1 else 0 for x in ytr])
    doubles_binary = torch.FloatTensor([1 if x[0] > 1 else 0 for x in yte])


    xtr = xtr.float().to(device)
    ytr = ytr.float().to(device)
    xval = xval.float().to(device)
    yval = yval.float().to(device)
    xte = xte.float().to(device)
    yte = yte.float().to(device)

    # kinda want to return like
    # A, DMMP, 1,0,0,0, 12,0,0,0,0, Chemcep Embed - to have everything on hand


    return [xtr, ytr], [xval, yval], [xte, yte], singles_binary, doubles_binary



def augbatch(xbatch, memlist = memlist, augprob = augprob, y = None):
    x1 = xbatch.clone()
    x2 = xbatch.clone()

    #x1s = []
    for i, (name, func) in enumerate(memlist):
        if random.uniform(0,1) < augprob:
            #print('x1 rolled below', augprob, 'applying', name, func)
            x1, y1 = func(x1, y)
            #x1s.append(name)
                
    #x2s = []
    for i, (name, func) in enumerate(memlist):
        if random.uniform(0,1) < augprob:
            #print('x1 rolled below', augprob, 'applying', name, func)
            x2, y2 = func(x2, y)
            #x2s.append(name)

    # for sens in x1[0].cpu():
    #     plt.plot(sens, alpha = 0.7)
    # plt.title(f'x1[0] vis, {x1s}')
    # plt.show()

    # for sens in x2[0].cpu():
    #     plt.plot(sens, alpha = 0.7)
    # plt.title(f'x2[0] vis, {x2s}')
    # plt.show()

    # print(braker)
    return x1, x2

def stdbatch(x, sec, start = 175):
    
    stop = 175 + int(sec*20)
    return x[:,:,start:stop].flatten(1)

def barlow_val(model, loader, criterion, sec = sec):
    # Getting nan's from testing data - verify integrity here?
    n_iter = 20
    critlist = []
    for _ in range(n_iter):
        
        
        for x,y in loader:
            x1, x2 = augbatch(x)
            x1 = stdbatch(x1, sec)
            x2 = stdbatch(x2, sec)

            #print('looking for nans', torch.sum(torch.isnan(x1)), torch.sum(torch.isnan(x2)))

            _, y1 = model(x1)
            _, y2 = model(x2)
            
            critlist.append(criterion(y1,y2))
        
    print('variance', torch.std(torch.FloatTensor(critlist)))
    return torch.mean(torch.FloatTensor(critlist))

def binarize(y):
    return torch.FloatTensor([1 if x[0] > 1 else 0 for x in y])

def supervised(barlow_model, supervised_model, xtr, ytr, xval, yval, xte, yte):
    # Get supervised performance given the Barlow embedder and some set
    ytr = binarize(ytr)
    yval = binarize(yval)
    yte = binarize(yte)
    
    model = SVC

    parameters = {
    "C":[0.0001, 0.0002, 0.0004, 0.001, 0.002, 0.004, 0.008], #.004
    "kernel":['linear', 'rbf', 'poly', 'sigmoid'],
    "shrinking":[True, False],
    "degree":[0,1,2,4,8],
    "tol":[2.2, 2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6] #1.0
    }

    mcc_scorer = metrics.make_scorer(matthews_corrcoef)
    supervised_criterion = matthews_corrcoef
    
    
    # barlow_model returns [representation, embedding] of input
    with torch.no_grad():
        tr_rep = barlow_model(barlow.stdbatch(xtr, sec))[0].cpu()
        val_rep = barlow_model(barlow.stdbatch(xval, sec))[0].cpu()
        te_rep = barlow_model(barlow.stdbatch(xte, sec))[0].cpu()

    # Supervised model implements fit_transform (fit x,y return yhat) and predict methods
    opt_model, (mu,sd) = augutils.CVopt(model, parameters, tr_rep, ytr, mcc_scorer, [])
    
    
    yhat = opt_model.predict(tr_rep)
    tr_loss = supervised_criterion(yhat, ytr)
    
    yval_hat = opt_model.predict(val_rep)
    val_loss = supervised_criterion(yval_hat, yval)
    
    yte_hat = opt_model.predict(te_rep)
    te_loss = supervised_criterion(yte_hat, yte)
        
    return opt_model, tr_loss, val_loss, te_loss


def default_parameters():

    parameters = {}
    parameters['rep_dim'] = 128
    parameters['emb_dim'] = 512

    parameters['learning_rate'] = 1e-5
    parameters['batch_size'] = 16
    parameters['n_training_iterations'] = 20

    parameters['barlow_lambda'] = 0.005

    parameters['criterion'] = metrics.matthews_corrcoef

    return parameters


class BarlowTwinsLoss(torch.nn.Module):

    def __init__(self, device, lambda_param=5e-3):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.device = device

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):
        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD

        #print(z_a_norm, z_b_norm)

        N = z_a.size(0)
        D = z_a.size(1)

        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N # DxD

        #print(sum(c.isnan()))
        # loss
        c_diff = (c - torch.eye(D,device=self.device)).pow(2) # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        loss = c_diff.sum()

        return loss

def train_barlow(xtr, parameters, val_dset = None, te_dset = None):

    # Dataset
    xtr = xtr.to(device).float()
    print('Training data shape', xtr.shape)
    print('Subsetting to', parameters['exposure_seconds'], 'seconds')
    print('Augmentations:', augprob)
    for element in memlist:
        print(element)

    tr_loader = torch.utils.data.DataLoader(list(xtr), batch_size = parameters['batch_size'])

    if val_dset is not None:
        val_loader = torch.utils.data.DataLoader(val_dset, batch_size = val_dset[0].shape[0])
    if te_dset is not None:
        test_loader = torch.utils.data.DataLoader(te_dset, batch_size = te_dset[0].shape[0])
    #

    num_iterations = parameters['n_training_iterations']
    val_stride = num_iterations // 20

    criterion = BarlowTwinsLoss(device = device, lambda_param = parameters['barlow_lambda'])
    mcc_scorer = metrics.make_scorer(parameters['criterion'])
    supervised_criterion = parameters['criterion']

    tr_losses = []
    tr_means = []
    val_losses = []
    test_losses = []

    strs = []
    svals = []
    stes = []

    model = barlow.Barlow(xtr.shape[1], parameters['rep_dim'], parameters['rep_dim'], 
        parameters['emb_dim'], parameters['emb_dim']).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = parameters['learning_rate'])

    it = 1
    while it < num_iterations:

        for x in tr_loader:
            x1, x2 = barlow.augbatch(x)

            for param in model.parameters():
                param.grad = None

            _, y1 = model(x1) # Ignore representation returns (lower-dim for representation)
            _, y2 = model(x2)
            loss = criterion(y1, y2) # Loss is taken between embeddings

            loss.backward()

            optimizer.step()

            it += 1

    return model


class SimpleBarlow(nn.Module):
    def __init__(self, in_size, r_size, e_size):
        super(SimpleBarlow, self).__init__()
        
        # Image -> Representation
        self.r1 = nn.Linear(in_size, r_size)
        self.rbn1 = nn.BatchNorm1d(self.r1.out_features)
        self.r2 = nn.Linear(r_size, r_size)
        
        # Representation -> Embedding
        self.e1 = nn.Linear(r_size, e_size)
        self.ebn1 = nn.BatchNorm1d(self.e1.out_features)
        self.e2 = nn.Linear(e_size, e_size)
        
        #self.bn = nn.BatchNorm1d
        
    def forward(self, x):
        # They use ResNet50. might have residuals, batchnorm, dropout....
        
        rep = self.r2(self.rbn1(self.r1(x)))
        
        emb = self.e2(self.ebn1(self.e1(rep)))
        
        return rep, emb

class Barlow(nn.Module):
    def __init__(self, in_size, r_h_size, r_size, e_h_size, e_size):
        super(Barlow, self).__init__()
        
        # Image -> Representation
        self.r1 = nn.Linear(in_size, r_h_size)
        self.rbn1 = nn.BatchNorm1d(self.r1.out_features)
        self.r2 = nn.Linear(r_h_size, r_h_size)
        self.rbn2 = nn.BatchNorm1d(self.r2.out_features)
        self.r3 = nn.Linear(r_h_size, r_size)
        
        # Representation -> Embedding
        self.e1 = nn.Linear(r_size, e_h_size)
        self.ebn1 = nn.BatchNorm1d(self.e1.out_features)
        self.e2 = nn.Linear(e_h_size, e_h_size)
        self.ebn2 = nn.BatchNorm1d(self.e2.out_features)
        self.e3 = nn.Linear(e_h_size, e_size)
        
        #self.bn = nn.BatchNorm1d
        
    def forward(self, x):
        # They use ResNet50. might have residuals, batchnorm, dropout....
        
        rep = F.relu(self.rbn1(self.r1(x)))
        rep = F.relu(self.rbn2(self.r2(rep)))
        rep = self.r3(rep)
        
        emb = F.relu(self.ebn1(self.e1(rep)))
        emb = F.relu(self.ebn2(self.e2(emb)))
        emb = self.e3(emb)
        
        return rep, emb