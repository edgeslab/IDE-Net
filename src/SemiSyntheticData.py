import pickle
import scipy.io as sio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import LatentDirichletAllocation
from torch_geometric.nn import aggr
from torch_geometric.utils import degree
import pickle
import os

class SemiSyntheticData(object):
    def __init__(self, data_path, num_features, data_name):
        self.path = data_path
        self.num_features = num_features
        self.data_name = data_name
        net = sio.loadmat(self.path)
        try:
            with open(f'../data/semisynthetic/{self.data_name}_features_{self.num_features}.pickle', 'rb') as file:
                self.features = pickle.load(file)
                print('Loaded preprocessed data')
        except:
            print('Could not find preprocessed features. Preparing it now')
            self.features = self.prepare(net)
        self.features = torch.Tensor(self.features)
        self.net = net['Network']
        row, col = net['Network'].nonzero()
        self.row = torch.LongTensor(row)
        self.col = torch.LongTensor(col)
        self.true_effects = None
        
    def reset_params(self, seed):
        np.random.seed(seed)
        # weights X to T
        W_xt = np.random.uniform(-3, 3, size=self.num_features)
        # weights X to Y
        W_xy = np.random.uniform(-3, 3, size=self.num_features)

        self.mask_xt = np.random.binomial(1, 0.6, size=self.num_features)
        self.mask_xy = np.random.binomial(1, 0.6, size=self.num_features)
        W_xt = self.mask_xt*W_xt
        W_xy = self.mask_xy*W_xy
        self.W_xt = torch.FloatTensor(W_xt)
        self.W_xy = torch.FloatTensor(W_xy)

    def prepare(self, net):
        X = net['Attributes'] # Bag of words of blogs
        lda = LatentDirichletAllocation(n_components=self.num_features, random_state=9)
        lda.fit(X)
        features = lda.transform(X)
        os.makedirs(f'../data/semisynthetic/', exist_ok=True)
        with open(f'../data/semisynthetic/{self.data_name}_features_{self.num_features}.pickle', 'wb') as file:
            pickle.dump(features, file)
        print('Saved preprocessed data')
        return features
    
    def sample_tau(self, vals):
        return np.random.choice(vals)

    def treatments(self, **kwargs):
        mean_aggr = aggr.MeanAggregation()
        peer_features = mean_aggr(self.features[self.col], self.row)
        tauC1 = self.sample_tau(kwargs.get('treatment_influence', [0.4]))
        ego_contrib = (1-tauC1)*self.W_xt*self.features
        peers_contrib = tauC1*self.W_xt*peer_features
        propensity = torch.sigmoid((ego_contrib+peers_contrib).sum(1))
        treatments = torch.where(propensity > 0.5, 1., 0.)
        return treatments

    def outcomes(self, seed, treatments, **kwargs):
        z_ind = np.setdiff1d(self.mask_xy.nonzero()[0], self.mask_xt.nonzero()[0])
        Z = self.features[:, z_ind]
        deg = degree(self.col, num_nodes=self.features.shape[0])
        quantile_aggr = aggr.QuantileAggregation(q=0.75)
        degq75 = quantile_aggr(deg[self.col].view(-1,1), self.row).view(-1)
        net_em = torch.where(deg > degq75, 1, 0)
        
        # exposure based on degree
        sum_aggr = aggr.SumAggregation()
        weight_tie = deg[self.col]
        
        # exposure based on similarity
        gamma = kwargs.get('gamma', 1.0)
        D_ij = ((Z[self.row]-Z[self.col])**2).sum(-1)
        weight_sim = (-gamma*D_ij).exp()
        
        # exposure based on number of mutual friends
        Er = torch.Tensor(self.net.toarray())
        mut_frns_Er1 = torch.einsum('ij,kj->ik', Er, Er)
        mut_frns_Er1.fill_diagonal_(0).pow_(0.5)
        weight_conn = mut_frns_Er1[self.row, self.col]
        
        np.random.seed(seed)
        tau_T = np.random.choice([-2.,-1.,1.,2.])
        tau_Z = self.sample_tau(kwargs.get('attr_em', [0]))
        tau_Er = self.sample_tau(kwargs.get('net_em', [0]))
        tau_X1_het_tie = self.sample_tau(kwargs.get('het_exposure', [0]))
        tau_X1_het_sim = self.sample_tau(kwargs.get('het_exposure_sim', [0]))
        tau_X1 = self.sample_tau(kwargs.get('exposure', [0]))
        tau_X1_het1_conn = self.sample_tau(kwargs.get('het_exposure_mut_frns', [0]))
        tau_X1_het1_div = self.sample_tau(kwargs.get('het_exposure_div', [0]))
        tau_X1_em = self.sample_tau(kwargs.get('exposure_em', [0]))
        
        print('gamma', gamma)
        print('tau_T',tau_T)
        print('tau_Z',tau_Z)
        print('tau_Er',tau_Er)
        print('tau_X1_het_tie',tau_X1_het_tie)
        print('tau_X1_het_sim',tau_X1_het_sim)
        print('tau_X1_het1_conn',tau_X1_het1_conn)
        print('tau_X1_het1_div',tau_X1_het1_div)
        print('tau_X1',tau_X1)
        print('tau_X1_em', tau_X1_em)
        
        weight = (tau_X1_het_tie*weight_tie) + (tau_X1_het_sim*weight_sim) + (tau_X1_het1_conn* weight_conn)
        num = sum_aggr((treatments[self.col]*weight).view(-1,1), self.row)
        deno = sum_aggr(weight[self.col].view(-1,1), self.row)
        exp = num.div(deno+1e-8).view(-1)
        
        
        
        outcome = tau_T*treatments
        outcome += (self.W_xy*self.features).sum(1) #confounders
        valZ = (tau_Z*self.W_xy[z_ind]*self.features[:,z_ind]).sum(1)
        outcome += treatments*valZ # node attrs effect modifiers
        outcome += (tau_Er*treatments*net_em) # net em
        outcome += (tau_X1*exp) # exposure similarity
        outcome += (tau_X1_em*exp*treatments)
        true_effects = tau_T + tau_Z*valZ + tau_Er*net_em + tau_X1_em*exp
        self.true_effects = true_effects
        return outcome
    
    

class BlogCatalogData(SemiSyntheticData):
    def __init__(self, data_path='../data/BC/BC0.mat', num_features=50):
        super().__init__(data_path, num_features, data_name='BC')
        

class FlickrData(SemiSyntheticData):
    def __init__(self, data_path='../data/Flickr/Flickr0.mat', num_features=50):
        super().__init__(data_path, num_features, data_name='Flickr')
       
        
        
        
    