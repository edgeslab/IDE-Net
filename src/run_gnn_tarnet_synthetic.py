import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from timeit import default_timer as timer

import networkx as nx
import time
import random
import os
from tqdm import tqdm
import random
import yaml
from datetime import datetime

from torch.nn import Linear
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.utils import degree
import torch_sparse
from torch_sparse import SparseTensor
from sklearn.model_selection import train_test_split
import flaml
from ray import tune
from torcheval.metrics import R2Score
from utils import wasserstein

os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['RAY_PICKLE_VERBOSE_DEBUG']='0'
os.environ['TUNE_MAX_PENDING_TRIALS_PG']='8'
os.environ['RAY_DISABLE_MEMORY_MONITOR']='1'

def set_seed(seed=1):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  
    torch.backends.cudnn.enabled = False
    
from nscm import *
from network_skeleton import NetworkSkeleton, NSCMGenerator
from nscm_functions import FunctionsSetup1

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def load_data(sk):
    A = torch.FloatTensor(sk.am('Er'))
    W = torch.FloatTensor(sk.am('Zr'))
    T = torch.FloatTensor(sk.nodes_df['X'].values)
    Y = torch.FloatTensor(sk.nodes_df['Y'].values)
    features = sorted(set(sk.nodes_df.columns) - set(['X', 'Y']))
    X = torch.FloatTensor(sk.nodes_df[features].values)
    edge_index = A.nonzero().t()
    row, col = edge_index
    edge_weights = W[row, col]
    edge = A[row, col]
    edge_attrs = torch.cat([edge.view(-1,1), edge_weights.view(-1,1)], dim=-1)
    return A, X, T, Y, edge_attrs

def normalizeY(Y):
    ymean, ystd = Y.mean(), Y.std()
    Y = (Y - ymean).div(ystd+1e-12)
    return Y, ymean, ystd

def denormalizeY(Y, ymean, ystd):
    return (Y*ystd) + ymean

def std_scaling(X):
    X = (X - X.mean(0,keepdim=True)).div(X.std(0,keepdim=True)+1e-12)
    return X

def minmax_scaling(X):
    min_val = X.min(dim=0)[0]
    max_val = X.max(dim=0)[0]
    deno = max_val - min_val
    return (X - min_val).div(deno+1e-12)


class MLPLayer(torch.nn.Module):
    def __init__(self, indim, outdim, num_layers=1, dropout=0., batchnorm=False):
        super(MLPLayer, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.emb = torch.nn.Linear(indim, outdim)
        self.mlp = torch.nn.ModuleList()
        self.bn = torch.nn.ModuleList()        
        for i in range(num_layers-1):
            self.mlp.append(Linear(outdim, outdim))
            if batchnorm:
                self.bn.append(torch.nn.BatchNorm1d(outdim))
            else:
                self.bn.append(torch.nn.Identity())
    
    def forward(self, x):
        emb = (self.emb(x))
        x = emb
        for i in range(self.num_layers-1):
            x = F.dropout(x, self.dropout, self.training)
            x = torch.tanh(self.bn[i](x))
            x = self.mlp[i](x)
            if i == self.num_layers-2:
                x = emb + x
        x = F.dropout(x, self.dropout, self.training)
        return x

class CANE_FeatureEmbedding(torch.nn.Module):
    """
    Feature embedding encoding [E].Z, [E,R].Z, [E,R,E].Z, and [E,R,E,R].Z
    """
    def __init__(self, node_dim, node_hidden, edge_dim, edge_hidden=None,
                 num_layers=1, dropout=0., bias=True, norm=True):
        super(CANE_FeatureEmbedding, self).__init__()
        assert num_layers >= 1
        assert node_hidden >= 2 and edge_hidden >=1
        torch.manual_seed(9)
        self.norm = norm
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = torch.nn.ReLU()

        ego_hidden = node_hidden//2
        peer_hidden = node_hidden - ego_hidden
        struct_hidden = ego_hidden
        if edge_hidden is None:
            edge_hidden = ego_hidden

        self.peer_embs = MLPLayer(node_dim+edge_dim, peer_hidden+edge_hidden, num_layers=num_layers)
        self.ego_emb = MLPLayer(node_dim, ego_hidden, num_layers=num_layers, dropout=dropout)
        self.edge_emb = MLPLayer(edge_dim, edge_hidden, num_layers=num_layers, dropout=dropout)
        self.hdim = node_hidden + 3*edge_hidden
        # self.hdim = node_hidden + edge_hidden

                
    def forward(self, x, A, edge_attrs=None):
        edge_index = A.nonzero().t()
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1]).t()
        h_ego = self.activation(self.ego_emb(x))
        if self.norm:
            norm = self.normalize(adj_t, column=False) # maybe more robust to network
        else:
            norm = adj_t
        peer_features = torch.cat([x[edge_index[1]], edge_attrs], dim=1)
        _,_,norm_val = norm.coo()
        h_peer = self.activation((adj_t.set_value(norm_val.view(-1,1)*self.peer_embs(peer_features))).sum(1))
        # h_peer = self.activation(norm.matmul(self.peer_embs(x)))        
        h_edge = self.activation(self.edge_emb(edge_attrs))
        h_edge = adj_t.set_value(h_edge).sum(1)
        # h_edge = norm.fill_diag(1).matmul(h_edge)
        h_edge = torch.cat([h_edge, norm.matmul(h_edge)], dim=1)
        
        out = torch.cat([h_ego, h_edge, h_peer], dim=1)
        return out, None
    
    def normalize(self, adj_t, column=False):
        deg = adj_t.sum(1) # degree
        if column:
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
            adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
            adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))
        else:
            deg_inv = deg.pow_(-0.5)
            deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
            adj_t = torch_sparse.mul(adj_t, deg_inv.view(-1, 1))

        return adj_t   

class VanillaHeterogeneousExposure(torch.nn.Module):
    """
    Exposure mapping that considers edge attributes and similarity of node embeddings 
    """
    def __init__(self, node_hidden,
                        edge_dim,
                        dropout=0., num_layers=1):
        super(VanillaHeterogeneousExposure, self).__init__()
        torch.manual_seed(9)
        self.hdim = 2*node_hidden + (edge_dim + 1)
        self.edge_emb = MLPLayer((edge_dim + 1), (edge_dim + 1), num_layers=num_layers, dropout=dropout)
        # self.sim_emb = torch.nn.Sequential(
        #     Linear(2*node_hidden, node_hidden),
        #     torch.nn.ReLU(),
        #     Linear(node_hidden, node_hidden),
        #     torch.nn.Sigmoid()
        # )
        # self.simW = torch.nn.Parameter(torch.ones(node_hidden))
        # self.hdim = (edge_dim + 1 + node_hidden)
    
    def forward(self, x, A, t, edge_attrs, AA):
        """
        TODO: Generalize to any edge attrs
        """
        row, col = A.nonzero().t()
        adj_t = SparseTensor(row=row, col=col).t()
        row, col, _ = adj_t.coo()
        t = t.view(-1,1)
        x_i, x_j = x[row], x[col]
        x_ij = (x_j - x_i)**2
        sim = torch.exp(-1*x_ij)
        # x_ij = torch.cat([x_i, x_j], dim=1)
        # sim = self.sim_emb(x_ij)
        aa = AA[row, col]
        edge_attrs = torch.cat([edge_attrs, aa.view(-1,1)], dim=1)
        edge_attrs = edge_attrs + F.relu(self.edge_emb(edge_attrs))
        edge_attrs = torch.cat([edge_attrs, x_j, sim], dim=1)
        out = self.wsum(adj_t, edge_attrs, t, row, col, scale=False)
        return out

    
    def wsum(self, adj_t, edge_attrs, t, row, col, scale=False):
        if scale:
            edge_attrs = minmax_scaling(edge_attrs) * (1-0.01) + 0.01
        edge_attrs_t = edge_attrs*t[col].view(-1,1)
        out = adj_t.set_value(edge_attrs_t).sum(1).div(adj_t.set_value(edge_attrs).sum(1)+1e-8)
        return out

    
    def softmax_agg(self, adj_t, edge_attrs, t, row, col):
        exp = edge_attrs - edge_attrs.max(0, keepdim=True)[0]
        exp = exp.exp()
        deno = adj_t.set_value(exp).sum(1) + 1e-8
        exp_t = exp.div(deno[row])*t[col]
        out = adj_t.set_value(exp_t).sum(1)
        return out
    
class VanillaGCNFeatureEmbedding(torch.nn.Module):
    def __init__(self, num_features, num_hidden, num_layers=1, dropout=0.):
        super(VanillaGCNFeatureEmbedding, self).__init__()
        assert num_layers >= 1
        torch.manual_seed(9)
        self.hdim = num_hidden
        self.dropout = dropout
        self.num_layers = num_layers
        self.activation = torch.nn.ReLU()
        self.gc = torch.nn.ModuleList([
            GCNConv(num_features, num_hidden)
        ])
        for i in range(num_layers-1):
            self.gc.append(GCNConv(num_hidden, num_hidden))
                
    def forward(self, features, A, edge_weights=None):
        edge_index = A.nonzero().t()
        # convert to sparse tensor for reproducibility. adj_t need for directed graphs
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1]).t()
        h = self.activation(self.gc[0](features, adj_t))
        h = F.dropout(h, self.dropout, self.training)
        for i in range(1, self.num_layers):
            h = self.activation(self.gc[i](h, adj_t))
            h = F.dropout(h, self.dropout, self.training)
        return h, None
    
class TLearner(torch.nn.Module):
    def __init__(self, num_features, num_hidden=32, out_layers=1, dropout=0., batchnorm=False):
        super(TLearner, self).__init__()
        torch.manual_seed(9)
        self.hid = num_hidden
        # self.hid = num_features
        self.out_layers = out_layers
        self.dropout = dropout
        # potential outcome
        self.embedding = MLPLayer(num_features, self.hid, num_layers=1, dropout=dropout, batchnorm=batchnorm)
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(self.hid, affine=True)
        if out_layers > 1:
            self.hid_t0 = MLPLayer(self.hid, self.hid, num_layers=out_layers-1, dropout=dropout, batchnorm=batchnorm)
            self.hid_t1 = MLPLayer(self.hid, self.hid, num_layers=out_layers-1, dropout=dropout, batchnorm=batchnorm)
        else:
            self.hid_t0 = torch.nn.Identity()
            self.hid_t1 = torch.nn.Identity()
        
        self.out_t0 = Linear(self.hid, 1)
        self.out_t1 = Linear(self.hid, 1)
        
    def forward(self, features):
        emb=features
        emb = self.relu(self.embedding(features))
        emb = torch.tanh(self.bn(emb))
        # emb = self.relu(self.bn(emb))

        y0 = self.hid_t0(emb)
        y1 = self.hid_t1(emb)
        y0_pred = self.out_t0(y0).view(-1)
        y1_pred = self.out_t1(y1).view(-1)
        return y0_pred, y1_pred, emb
    
class GNN_TLearner(torch.nn.Module):
    def __init__(self, num_features, f_hid, e_hid=8, in_layers=1, out_layers=1,
                 dropout=0, exposure_type=0, vanilla=False, edge_dim=2, clip=0, motifs=None):
        super().__init__()
        torch.manual_seed(9)
        dp = dropout
        # dropout = 0.
        self.vanilla = vanilla
        if vanilla:
            self.featuremap = VanillaGCNFeatureEmbedding(num_features, f_hid,
                                                  num_layers=in_layers, dropout=dropout)
        else:
            self.featuremap = CANE_FeatureEmbedding(node_dim=num_features, node_hidden=f_hid,
                                        edge_dim=edge_dim, edge_hidden=e_hid,
                                        num_layers=in_layers, dropout=dropout)

        self.f_hid = f_hid
        self.e_hid = e_hid
        self.exposure_type = exposure_type
        if exposure_type == 2:
            self.exposuremap = VanillaHeterogeneousExposure(node_hidden=self.featuremap.hdim,
                                                            edge_dim=edge_dim,
                                                            dropout=dropout, num_layers=in_layers
                                                           )
            self.hid = self.featuremap.hdim + self.exposuremap.hdim
        elif exposure_type == 1:
            if motifs is None:
                self.hid = self.featuremap.hdim + 1
            else:
                self.hid = self.featuremap.hdim + motifs
        else:
            self.hid = self.featuremap.hdim
        self.out_layers = out_layers
        dropout = dp
        # potential outcome
        self.tlearner = TLearner(self.hid, self.f_hid, dropout=dropout,
                                 out_layers=self.out_layers)
        self.AA = None
                
        if clip > 0:
            modules = [self.tlearner]
            for module in modules:
                for p in module.parameters():
                    p.register_hook(lambda grad: torch.clamp(grad, -clip, clip))
        
    def forward(self, A, features, treatments, edge_weights=None, external=None):
        f_emb, attn_wts = self.featuremap(features, A, edge_weights)
        f_emb = F.relu(f_emb)
        if self.exposure_type == 2:
            if self.AA is None:
                self.AA = torch.einsum('ij,kj->ik', A, A)*A
            if external is not None:
                e_emb = external.to(A.device)
            else:
                e_emb = self.exposuremap(f_emb, A, treatments, edge_weights, self.AA)
            rep = torch.cat([f_emb, e_emb], dim=1)
        elif self.exposure_type == 1:
            deg = A.sum(dim=1) + 1e-8
            if external is not None:
                e_emb = external.to(A.device)
            else:
                e_emb = (A*treatments).sum(dim=1).div(deg)
                e_emb = e_emb.view(-1,1)
            rep = torch.cat([f_emb, e_emb], dim=1)
        else:
            rep = f_emb
        y0_pred, y1_pred, rep = self.tlearner(rep)
        return y0_pred, y1_pred, rep


def tune_and_train_gnn_model(config, A=None, X=None, T=None,
                   Y=None, W=None, motifs=None, checkpoint_dir=None, enable_tune=True):
    warnings.warn = warn
    model = GNN_TLearner(X.shape[1],
                         config.get('f_hid', 32),
                         exposure_type=config.get('exposure_type', 2),
                         out_layers=config.get('out_layers', 1),
                         in_layers=config.get('in_layers', 1),
                         dropout=config.get('dropout', 0.),
                         vanilla=config.get('vanilla', False),
                         edge_dim=W.shape[1],
                         e_hid=config.get('e_hid', 4),
                         clip=config.get('clip', 0),
                         motifs=motifs.shape[1] if motifs is not None else None,
                        )
    # print(model)
    model = model.to(X.device)
    epochs = config.get('epochs', 300)
    l1_lambda = config.get('l1_lambda', 0.0)
    l2_lambda = config.get('l2_lambda', 1e-5)
    alpha_dist = config.get('alpha_dist', 0)
    parameters = [{'params':model.featuremap.parameters()}]
    if config.get('exposure_type', 2) == 2:
        parameters.append({'params':model.exposuremap.parameters()})
    lrEnc = config.get('lr', 1e-1)
    lrEst = config.get('lr_est', lrEnc)
    print(f'LR_ENC: {lrEnc}, LR_EST: {lrEst}')
    optimizerEnc = torch.optim.Adam(parameters,
                                 lr=lrEnc, # Slower learning rate
                                 weight_decay=l2_lambda) 
    optimizerEst = torch.optim.Adam([{'params': model.tlearner.parameters()}],
                                    lr=lrEst, # Faster learning rate
                                    weight_decay=l2_lambda)
    schedulerEnc = torch.optim.lr_scheduler.StepLR(optimizerEnc,
                                          step_size=config.get('lr_step_size',50),
                                          gamma=config.get('lr_step_gamma', 0.5))
    schedulerEst = torch.optim.lr_scheduler.StepLR(optimizerEst,
                                      step_size=config.get('lr_step_size',50),
                                      gamma=config.get('lr_step_gamma', 0.5))
    criterion = torch.nn.MSELoss()  # Define loss criterion.
    r2score = R2Score()
    train_index, val_index = train_test_split(np.arange(X.shape[0]), 
                                              test_size=config.get('num_val', 0.2),
                                              stratify=T.cpu().numpy(),
                                              random_state=9)
    data = Data(x=X,y=Y, train_index=train_index, val_index=val_index).to(A.device)
    best_loss = float('inf')
    # early stopping
    eval_interval = config.get('eval_interval', 100)
    max_patience = config.get('max_patience', 10)
    patience = 0
    for epoch in range(epochs):
        model.train()
        optimizerEnc.zero_grad()  # Clear gradients.
        optimizerEst.zero_grad()
        y0_pred, y1_pred, rep = model(A, X, T, W, external=motifs)  # Perform a single forward pass.
        y_pred = torch.where(T > 0, y1_pred, y0_pred)
        pred_loss = criterion(y_pred[data.train_index], Y[data.train_index])  # Compute the loss solely based on the training nodes.

        l1_loss = 0.
        for param in model.tlearner.parameters():
            l1_loss += torch.abs(param).mean()
        if alpha_dist > 0:
            rep_t1 = rep[data.train_index][(T[data.train_index] > 0).nonzero()]
            rep_t0 = rep[data.train_index][(T[data.train_index] < 1).nonzero()]
            dist, _ = wasserstein(rep_t1, rep_t0, cuda=A.device.type!='cpu')
        else:
            dist = 0.
        if epoch >= 150:
            smooth_coeff = torch.exp(-3*torch.var(y1_pred-y0_pred)).detach()
        else:
            smooth_coeff = 0.
        loss = pred_loss  + alpha_dist * dist + l1_lambda*smooth_coeff*torch.var(y1_pred-y0_pred) #+ l1_lambda*l1_loss F.sigmoid(model.reg)
        loss.backward()  # Derive gradients.
        optimizerEnc.step()  # Update parameters based on gradients.
        optimizerEst.step()
        schedulerEnc.step() # Update learning rate scheduler
        schedulerEst.step()
        with torch.no_grad():
            model.eval()
            y0_pred, y1_pred, rep = model(A, X, T, W, external=motifs)
            y_pred = torch.where(T > 0, y1_pred, y0_pred)
            loss_val = criterion(y_pred[data.val_index], Y[data.val_index])
            rep_t1 = rep[data.val_index][(T[data.val_index] > 0).nonzero()]
            rep_t0 = rep[data.val_index][(T[data.val_index] < 1).nonzero()]
            val_dist, _ = wasserstein(rep_t1, rep_t0, cuda=A.device.type!='cpu')
            # loss_val += l1_loss
            r2_val = r2score.update(y_pred[data.val_index].view(-1,1),
                                    Y[data.val_index].view(-1,1)).compute()
        if loss_val < best_loss and epoch > 50:
            best_loss = loss_val
            patience = 0
        else:
            patience += 1

        if epoch % eval_interval == 0:
            if enable_tune:
                tune.report(loss=loss_val.item(), r2=r2_val.item(),
                        dist=val_dist.item(), regloss=(loss_val.item()+val_dist.item()))
            else:
                print(f'''Epoch: {epoch}, Train loss: {loss.item()}, Dist: {dist},
                Val loss: {loss_val.item()}, Val Dist: {val_dist.item()}, Val r2: {r2_val.item()}''')

        if patience == max_patience and epoch>149:
            break
    if not enable_tune:
        print(f'''Epoch: {epoch}, Train loss: {loss.item()}, Dist: {dist},
        Val loss: {loss_val.item()}, Val Dist: {val_dist}, Val r2: {r2_val.item()}''')
        return model
    else:
        tune.report(loss=loss_val.item(), r2=r2_val.item(),
        dist=val_dist.item(), regloss=(loss_val.item()+val_dist.item()))
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(model.state_dict(), path)

def train(A, X, T, Y, W, exposure_type, epochs=300, lr=1e-1, lr_est=0.05, lr_decay=0.5,
          lr_step=100, weight_decay=1e-5, vanilla=False, num_val=0.2, f_hid=32, in_layers=2, max_patience=300, e_hid=4, out_layers=1, dropout=0., alpha=0., clip=2., normY=False, verbose=0, exp_name=None, motifs=None, reg=1):
    l1params = reg*np.array([0.1, 1.0])
    l1params = np.unique(l1params)
    # l1params = np.array([0])    
    params = {
        'l1_lambda': tune.grid_search(l1params.tolist()),
        'lr':lr,#tune.grid_search([lr]),
        'lr_est': lr_est,#tune.grid_search([lr_est]),
        'l2_lambda': weight_decay,
        'lr_step_size':lr_step,
        'lr_step_gamma':lr_decay,
        'clip': clip,#tune.grid_search([clip]),
        'max_patience':max_patience, 'epochs':epochs,
        'out_layers':out_layers, 'in_layers':in_layers,#tune.grid_search([in_layers]),
        'f_hid': f_hid, 'e_hid': e_hid, 'dropout': dropout,
        'num_val':num_val, 'alpha_dist': alpha,
        'exposure_type':exposure_type, 'vanilla': vanilla
    }
    num_samples=1
    time_budget_s = 3000
    trials = tune.run(tune.with_parameters(tune_and_train_gnn_model,
                                         A=A, X=X, T=T, Y=Y, W=W, motifs=motifs),
                    config=params,
                    # scheduler='medianstopping',
                    # search_alg='blendsearch',
                    metric="loss",
                    mode="min",
                    resources_per_trial={'cpu':4, 'gpu':0.16, 'memory':2*1024*1024*1024},
                    time_budget_s=time_budget_s,
                    verbose=verbose,
                    num_samples=num_samples,
                    local_dir='/home/sadhik9/ray_results/tune_and_train_gnn_model',
                    name=exp_name,
                    max_concurrent_trials=4,
                   )
    return trials

def evaluate(A, X, T, W, Y, ymean, ystd, trials, sk, metric, measure='min', normY=True, isolated=False, motifs=None):
    best_trial = trials.get_best_trial(metric, measure, "last")
    best_cfg = best_trial.config
    print(best_trial.checkpoint.value)
    # print(best_cfg)
    checkpoint_value = getattr(best_trial.checkpoint, "dir_or_data", None) or best_trial.checkpoint.value
    checkpoint_path = os.path.join(checkpoint_value, "checkpoint")
    model_state = torch.load(checkpoint_path)
    best_model = GNN_TLearner(X.shape[1],
                         best_trial.config.get('f_hid', 32),
                         exposure_type=best_trial.config.get('exposure_type', 2),
                         out_layers=best_trial.config.get('out_layers', 1),
                         in_layers=best_trial.config.get('in_layers', 1),
                         dropout=best_trial.config.get('dropout', 0.),
                         vanilla=best_trial.config.get('vanilla', False),
                         edge_dim=W.shape[1],
                         e_hid=best_trial.config.get('e_hid', 4),
                         motifs=motifs.shape[1] if motifs is not None else None,
                        )
    best_model = best_model.to(Y.device)
    # print(best_model)
    best_model.load_state_dict(model_state)
    with torch.no_grad():
        if isolated:
            if best_model.exposure_type == 2:
                zero_exp = torch.zeros((len(A), best_model.exposuremap.hdim))
                external = zero_exp
            elif best_model.exposure_type == 1:
                zero_exp = torch.zeros((len(A), 1))
                external = zero_exp
            else:
                external = motifs
        else:
            external = motifs
        
        best_model.eval()
        y_t0, y_t1, _ = best_model(A,X,T,W, external=external)
        if normY:
            y_t0 = denormalizeY(y_t0, ymean, ystd)
            y_t1 = denormalizeY(y_t1, ymean, ystd)
    err_ate = np.abs(sk.true_effects.mean() - (y_t1.mean() - y_t0.mean()).cpu().numpy())
    err_pehe = np.sqrt(np.mean((sk.true_effects - (y_t1 - y_t0).cpu().numpy())**2))
    err_avg = np.sqrt(np.mean((sk.true_effects - (y_t1.mean() - y_t0.mean()).cpu().numpy())**2))
    return err_ate, err_pehe, err_avg, best_cfg


def get_causal_motifs(sk):
    assignments = sk.nodes_df['X']
    G = sk.net
    am = sk.am('Er')
    data = []
    for i in tqdm(range(len(G))):
        neighbor = len(G[i])
        bb_1 = np.sum([assignments[j] for j in G[i]])
        bb_0 = neighbor - bb_1
        bbb_0 = 0
        bbb_1 = 0
        bbb_2 = 0
        bbn_0 = 0
        bbn_1 = 0
        bbn_2 = 0
        for j in G[i]:
            for k in G[i]:
                if k > j:
                    if am[j,k]==1:
                        if assignments[j] + assignments[k] == 0:
                            bbb_0 += 1
                        elif assignments[j] + assignments[k] == 1:
                            bbb_1 += 1
                        else:
                            bbb_2 += 1
                    else:
                        if assignments[j] + assignments[k] == 0:
                            bbn_0 += 1
                        elif assignments[j] + assignments[k] == 1:
                            bbn_1 += 1
                        else:
                            bbn_2 += 1
        bb = max(bb_0 + bb_1, 1)
        bbb = max(bbb_0 + bbb_1 + bbb_2, 1)
        bbn = max(bbn_0 + bbn_1 + bbn_2, 1)
        data.append([bb_0/bb, bb_1/bb, bbb_0/bbb, bbb_1/bbb, bbb_2/bbb, bbn_0/bbn, bbn_1/bbn, bbn_2/bbn])
    return torch.FloatTensor(data)

def main():
    parser = argparse.ArgumentParser(description='Run GNN-based Heterogeneous Network Effects Experiments')
    parser.add_argument('--estimator', type=str, default='INE-TARNet',
                        choices=['INE_TARNet', 'INE_TARNet_ONLY', 'INE_TARNet_INT', 'INE_TARNet_MOTIFS',
                                 'INE_CFR', 'INE_CFR_INT', 'INE_CFR_MOTIFS',
                                 'GCN_TARNet', 'GCN_TARNet_INT', 'GCN_TARNet_MOTIFS',
                                 'GCN_CFR', 'GCN_CFR_INT', 'GCN_CFR_MOTIFS'
                                ],
                        help='Estimator')
    parser.add_argument('--config', type=str, help='YAML file with config for experiments')
    # parser.add_argument('--exposure', type=str, choices=['hom', 'het', 'both', 'no'], default='both',
                       # help='Experiment for homogeneous or heterogeneous exposure') 
    # parser.add_argument('--vanilla', type=str, choices=['yes', 'no'], default='no')
    parser.add_argument('--maxiter', type=int, default=300, help='Maximum epochs')
    parser.add_argument('--val', type=float, default=0.2, help='Fraction of nodes used for validation and early stopping')
    parser.add_argument('--lr', type=float, default=0.02, help='Learning rate for encoder')
    parser.add_argument('--lrest', type=float, default=0.2, help='Learning rate for learner')
    parser.add_argument('--lrstep', type=int, default=50, help='Change LR after N steps')
    parser.add_argument('--lrgamma', type=float, default=0.5, help='lr=gamma*lr after N steps')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Regularization')
    parser.add_argument('--clip', type=float, default=3, help='Clip gradient')
    parser.add_argument('--max_patience', type=int, default=300, help='Early stopping if loss not improved')
    parser.add_argument('--fdim', type=int, default=32, help='Hidden layer (node embedding) dimension')
    parser.add_argument('--edim', type=int, default=4, help='Hidden layer (edge embedding) dimension')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout')  
    parser.add_argument('--inlayers', type=int, default=2, help='MLP layers for feature/exposure mapping')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha like CFR estimator')
    parser.add_argument('--normY', type=int, default=0, choices=[0,1], help='Normalize Y')
    parser.add_argument('--verbose', type=int, default=0, help='Verbose for parameters tuning')
    parser.add_argument('--isolated', type=int, default=0, choices=[0,1], help='Evaluate isolated direct effects')
    # parser.add_argument('--motifs', type=int, default=0, choices=[0,1], help='Get causal network motifs')
    parser.add_argument('--reg', type=int, default=1, choices=[0,1], help='Variance smoothing regularization')
    parser.add_argument('--outfolder', type=str, default='../results', help='Folder to output records')
    
    args = parser.parse_args()
    
    print(f'Estimator: {args.estimator}')
    exposure_types = [0]
    use_motifs = 0
    vanilla = True
    if args.estimator.startswith("INE_"):
        exposure_types = [2,1]
        vanilla = False
    if args.estimator.endswith("_ONLY"):
        exposure_types = [2]
        vanilla = False
        
    if args.estimator.endswith("_INT"):
        exposure_types = [1]
    if args.estimator.endswith("_MOTIFS"):
        exposure_types = [1]
        use_motifs = 1
    
    if "_TARNet" in args.estimator:
        args.alpha = 0.
    else:
        assert args.alpha > 0
        
    isolated = (args.isolated == 1)
    
    config = None
    exp_name = args.config.split('/')[-1].replace('.yaml', '')
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)
        print(config)
    model = config['model']
    nscm = NSCM(model)
    print(nscm.model)    
    seeds = config['network']['seeds']
    seed_start = config['network'].get('seed_start', 0)
    net_types = config['network']['skeleton_type']
    Ns = config['network']['N']
    ms = config['network']['m']
    ps = config['network']['p']
    function_config = config['functions']
    experiment = config['experiment']
    now = datetime.now()
    suffix = now.strftime('%Y-%m-%d-%H-%M')
    out_file = os.path.join(args.outfolder, f'records_{args.estimator}_{exp_name}_{suffix}.csv')
    records = []
    normX, normY = True, args.normY>0
    print(f"========normX: {normX}, normY: {normY}=========")
    for seed in tqdm(range(seed_start, seed_start+seeds)):
        for net_type in net_types:
            for N in Ns:
                for m,p in zip(ms, ps):
                    # set_seed(seed)
                    # sk = NetworkSkeleton(nscm, net_type, seed=seed, N=N, m=m, p=p)
                    for param, values in experiment.items():
                        for value in values:
                            set_seed(seed)
                            sk = NetworkSkeleton(nscm, net_type, seed=seed, N=N, m=m, p=p)
                            function_config['taus'][param] = value
                            functions = FunctionsSetup1(seed=seed, **function_config)
                            sk.instance(functions)
                            A, X, T, Y, W = load_data(sk)
                            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                            A, X, T, Y, W = A.to(device), X.to(device), T.to(device), Y.to(device), W.to(device)
                            if use_motifs > 0:
                                motifs = get_causal_motifs(sk)
                                motifs = motifs.to(device)
                            else:
                                motifs = None
                            if normX:
                                X = std_scaling(X)
                                W = std_scaling(W)
                                W[:,0] = 1. #first column edge existence
                            if normY:
                                Y, ymean, ystd = normalizeY(Y)
                            else:
                                ymean, ystd = None, None
                                
                            for exposure_type in exposure_types:
                                trials = train(A, X, T, Y, W,
                                            exposure_type=exposure_type,
                                            epochs=args.maxiter,
                                            lr=args.lr,
                                            lr_est=args.lrest,
                                            weight_decay=args.weight_decay,
                                            num_val=args.val,
                                            f_hid=args.fdim,
                                            e_hid=args.edim,
                                            vanilla=vanilla,
                                            normY=normY,
                                            dropout=args.dropout,
                                            in_layers=args.inlayers,
                                            alpha=args.alpha,
                                            lr_decay=args.lrgamma,
                                            lr_step=args.lrstep,
                                            clip=args.clip,
                                            verbose=args.verbose,
                                            exp_name=exp_name,
                                            motifs=motifs,
                                            reg=args.reg
                                            )
                                record = {
                                    'net_type': net_type, 'N': N, 'm': m, 'p': p, 'seed':seed,
                                    # 'learner': learner, 'adjustment': adjustment,
                                    param: value[0] if type(value)==list else value,
                                    'estimator': args.estimator,
                                    'exposure_type': exposure_type,
                                }
                                for metric in ['loss', 'r2']:
                                    measure = 'max' if metric=='r2' else 'min'
                                    err_ate, err_pehe, err_avg, best_cfg = evaluate(A, X, T, W, Y, ymean, ystd, trials, sk, metric, measure, normY=normY, isolated=isolated, motifs=motifs)
                                    record[f'err_ate_{metric}'] = err_ate
                                    record[f'err_pehe_{metric}'] = err_pehe
                                    record[f'err_avg_{metric}'] = err_avg
                                    record[f'cfg_{metric}'] = best_cfg

                                print(record)
                                records.append(record)
                    recordDf = pd.DataFrame(records)
                    recordDf.to_csv(out_file, index=False)

if __name__=='__main__':
    torch.use_deterministic_algorithms(True, warn_only=True)
    main() 
                            