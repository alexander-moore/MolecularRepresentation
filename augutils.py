# Utils for augmentation
import itertools
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import matthews_corrcoef

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


from inspect import getmembers, isfunction
import augmentations
memlist = getmembers(augmentations, isfunction)
#augprob = 0.20 # maybe this augprob gets set to some function of length(memlist), where the expected value of augs is like 2 
augprob = 1.5/len(memlist) # EV is 1.5
print('Augmentation probability', augprob)

sec = 5


def bin_list(n):
    return list(itertools.product([0, 1], repeat=n))

def CVopt(model, parameters, xtr, ytr, opt_scorer, metrics_list):
    cv = GridSearchCV(model(), parameters, scoring = opt_scorer, cv = 5, n_jobs = 8)

    print(xtr.shape, ytr.shape)
    #print(xtr, ytr)
    cv.fit(xtr, ytr)

    #display(cv)
    print(cv.best_params_)
    #print('Cross-validation mean holdout Matthews Correlation Coefficient')
    #print(cv.cv_results_['mean_test_score'][cv.best_index_], '+/-',cv.cv_results_['std_test_score'][cv.best_index_])

    opt_model = model(**cv.best_params_)
    opt_model.fit(xtr, ytr)

    #yhat_te = opt_model.predict(xte)

    #metric_list = [met(yte, yhat_te) for met in metrics_list]

    return opt_model, (cv.cv_results_['mean_test_score'][cv.best_index_], cv.cv_results_['std_test_score'][cv.best_index_])

def perform_bin_aug(bin_aug, x, y):
    from inspect import getmembers, isfunction
    import augmentations
    auglist = getmembers(augmentations, isfunction)

    augnames = []
    for augon, aug in zip(bin_aug, auglist):
        if augon > 0:
            augnames.append(aug[0])
            x,y = aug[1](x,y)

    return (x,y), augnames

def augbatch(xbatch, memlist = memlist, augprob = augprob, y = None):
    x = xbatch.clone()

    #x1s = []
    for i, (name, func) in enumerate(memlist):
        if random.uniform(0,1) < augprob:
            #print('x1 rolled below', augprob, 'applying', name, func)
            x, y = func(x, y)
            #x1s.append(name)

    # for sens in x1[0].cpu():
    #     plt.plot(sens, alpha = 0.7)
    # plt.title(f'x1[0] vis, {x1s}')
    # plt.show()

    # for sens in x2[0].cpu():
    #     plt.plot(sens, alpha = 0.7)
    # plt.title(f'x2[0] vis, {x2s}')
    # plt.show()

    # print(braker)
    return x,y

def eval_distribution(model, x, y, sec, metric = matthews_corrcoef, method = None, barlow = None, lookup_embs = None):
    # Evaluate a model over the distribution of testing augmentations given by augmentations.py
    # Model must implement a forward method

    # Allow this to account for devise latent by composing emb_to_class with matthews
    from inspect import getmembers, isfunction
    import augmentations

    memlist = getmembers(augmentations, isfunction)
    augnames = [x[0] for x in memlist]

    n_aug = len(memlist)
    binary_list = bin_list(n_aug)

    scores = []

    with torch.no_grad():
        for te_bin_aug in binary_list:
            (xaug, yaug), augnames = perform_bin_aug(te_bin_aug, x, y)

            xaug = xaug[:,:,175:int(175+sec*20)].flatten(1)

            if barlow is not None:
                xaug, _ = barlow(xaug.float())
                xaug = xaug#.cpu(

            if method == 'sklearn':
                xaug = xaug.cpu()

            yhat = model.predict(xaug)

            if method == 'emb_to_metric':
                score = metric(yhat.cpu(), lookup_embs, y)
            #print(yaug.shape, yhat.shape)
            if method == 'sklearn':
                score = metric(yaug, yhat)


            elif method == 'pytorch':
                score = metric(yhat.cpu(), yaug.cpu())
            scores.append(score)

    return scores, torch.mean(torch.FloatTensor(scores)), torch.std(torch.FloatTensor(scores))


class Predictor(nn.Module):
        def __init__(self, in_size, out_size, model_size):
            super(Predictor, self).__init__()
            self.fc1 = nn.Linear(in_size, model_size)
            self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features)
            self.fc3 = nn.Linear(self.fc2.out_features, out_size)

            self.out_activ = nn.Identity(out_size) if out_size > 1 else torch.sigmoid
        
        # forward method
        def forward(self, x):
            x = F.leaky_relu(self.fc1(x), 0.2)
            x = F.leaky_relu(self.fc2(x), 0.2)
            return self.out_activ(self.fc3(x))
        
        def predict(self, x):
            return self.forward(x)