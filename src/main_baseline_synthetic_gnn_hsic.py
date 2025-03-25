import torch
import torch.nn.functional as F

import time
import random
import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml
from datetime import datetime

os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from sklearn.model_selection import train_test_split
def set_seed(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  
    # torch.backends.cudnn.enabled = False
    
torch.use_deterministic_algorithms(True, warn_only=True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed()%2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv, GraphConv
# https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GraphConv.html#torch_geometric.nn.conv.GraphConv
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.utils import degree
import torch_sparse
from torch_sparse import SparseTensor
from torcheval.metrics import R2Score
from utils import wasserstein

from nscm import *
from network_skeleton import NetworkSkeleton, NSCMGenerator
from nscm_functions import FunctionsSetup1

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def load_data(sk):
    A = torch.FloatTensor(sk.am('Er'))
    T = torch.FloatTensor(sk.nodes_df['X'].values)
    Y = torch.FloatTensor(sk.nodes_df['Y'].values)
    features = sorted(set(sk.nodes_df.columns) - set(['X', 'Y']))
    X = torch.FloatTensor(sk.nodes_df[features].values)
    return A, X, T, Y

def std_scaling(X):
    X = (X - X.mean(0,keepdim=True)).div(X.std(0,keepdim=True)+1e-12)
    return X

class GNN_HSIC(torch.nn.Module):
    def __init__(self, x_dim, h_dim=32, g_dim=32, n_out=1, dropout=0.1, encoder='gcn'):
        super(GNN_HSIC, self).__init__()
        torch.manual_seed(9)
        self.h_dim = h_dim
        self.n_out = n_out
        self.g_dim = g_dim
        self.dropout = dropout
        self.encoder = encoder

        self.phi_x = torch.nn.Sequential(Linear(x_dim, self.h_dim), torch.nn.ReLU())
        
        if self.encoder == '1-gnn':
            self.gnn = GraphConv(self.h_dim, self.g_dim)
        else:
            self.gnn = GCNConv(self.h_dim, self.g_dim)

        self.y_rep_dim = self.h_dim + self.g_dim + 1

        # potential outcome
        self.out_t00 = torch.nn.ModuleList([Linear(self.y_rep_dim, self.y_rep_dim) for i in range(self.n_out)])
        self.out_t10 = torch.nn.ModuleList([Linear(self.y_rep_dim, self.y_rep_dim) for i in range(self.n_out)])
        self.out_t01 = Linear(self.y_rep_dim, 1)
        self.out_t11 = Linear(self.y_rep_dim, 1)

    def forward(self, X, A, T):
        phi_x = self.phi_x(X)
        phi_x_t = torch.mul(T.view(-1, 1), phi_x)
        row, col = A.nonzero().t()
        adj_t = SparseTensor(row=row, col=col).t()

        rep_gnn = F.relu(self.gnn(phi_x_t, adj_t))
        rep_gnn = F.dropout(rep_gnn, self.dropout, training=self.training)
        
        z = adj_t.set_value(T[col]).sum(1) / adj_t.sum(1)
        rep_post = torch.cat([phi_x, rep_gnn, z.view(-1,1)], dim=1)

        # potential outcome
        if self.n_out == 0:
            y00 = rep_post
            y10 = rep_post
        for i in range(self.n_out):
            y00 = F.relu(self.out_t00[i](rep_post))
            y10 = F.relu(self.out_t10[i](rep_post))

        y0_pred = self.out_t01(y00).view(-1)
        y1_pred = self.out_t11(y10).view(-1)

        return y0_pred, y1_pred, rep_post
    
def train_gnn_hsic(config, X, A, T, Y):
    lr = config.get('lr', 0.01)
    l2_lambda = config.get('l2_lambda', 1e-5)
    emb_dim = config.get('emb_dim', 32)
    rep_dim = config.get('rep_dim', emb_dim)
    out_dim = config.get('out_dim', 1)
    alpha_dist = config.get('alpha_dist', 0.5)
    encoder = config.get('encoder', 'gcn')
    N, in_dim = X.shape
    device = X.device
    model = GNN_HSIC(in_dim, h_dim=emb_dim, g_dim=rep_dim, n_out=out_dim, encoder=encoder).to(device)
    clip = config.get('clip', 0)
    if  clip > 0:
        modules = [model]
        for module in modules:
            for p in module.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, -clip, clip))
                
    optimizer = torch.optim.Adam(model.parameters(),
                             lr=lr, 
                             weight_decay=l2_lambda)
    criterion = torch.nn.MSELoss()  # Define loss criterion.
    r2score = R2Score()
    train_index, val_index = train_test_split(np.arange(X.shape[0]), 
                                          test_size=config.get('num_val', 0.2),
                                          stratify=T.cpu().numpy(),
                                          random_state=9)
    data = Data(x=X,y=Y, train_index=train_index, val_index=val_index).to(A.device)
    best_loss = float('inf')

    epochs = config.get('epochs', 10)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y0_pred, y1_pred, rep = model(X, A, T)
        y_pred = torch.where(T > 0, y1_pred, y0_pred)
        pred_loss = criterion(y_pred[data.train_index], Y[data.train_index])  # Compute the loss solely based on the training nodes.
        if alpha_dist > 0:
            rep_t1 = rep[data.train_index][(T[data.train_index] > 0).nonzero()]
            rep_t0 = rep[data.train_index][(T[data.train_index] < 1).nonzero()]
            dist, _ = wasserstein(rep_t1, rep_t0, cuda=A.device.type!='cpu')
        else:
            dist = 0.

        loss = pred_loss  + alpha_dist * dist
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        
        # with torch.no_grad():
        #     model.eval()
        #     y0_pred, y1_pred, rep = model(X, A, T)
        #     y_pred = torch.where(T > 0, y1_pred, y0_pred)
        #     loss_val = criterion(y_pred[data.val_index], Y[data.val_index])
        #     rep_t1 = rep[data.val_index][(T[data.val_index] > 0).nonzero()]
        #     rep_t0 = rep[data.val_index][(T[data.val_index] < 1).nonzero()]
        #     val_dist, _ = wasserstein(rep_t1, rep_t0, cuda=A.device.type!='cpu')
        #     # loss_val += l1_loss
        #     r2_val = r2score.update(y_pred[data.val_index].view(-1,1),
        #                             Y[data.val_index].view(-1,1)).compute()
        #     if epoch % config.get('eval_interval', 300) == 0:
        #         print(f'''Epoch: {epoch}, Train loss: {loss.item()}, Dist: {dist},
        #         Val loss: {loss_val.item()}, Val Dist: {val_dist.item()}, Val r2: {r2_val.item()}''')
                
    return model  

def main():
    parser = argparse.ArgumentParser(description='Baseline GNN-HSIC implementation')
    parser.add_argument('--config', type=str, help='YAML file with config for experiments')
    parser.add_argument('--estimator', type=str, default='1GNN_HSIC', choices=['1GNN_HSIC', 'GCN_HSIC'], help='Name of the estimator')
    parser.add_argument('--lr', type=float, default=0.1, help="Learning rate")
    parser.add_argument('--fdim', type=int, default=32, help='Hidden layer (node embedding) dimension')
    parser.add_argument('--clip', type=float, default=3, help='Clip gradient')
    parser.add_argument('--epochs', type=int, default=300, help='Number of Epochs')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha like CFR estimator for balancing')
    parser.add_argument('--outfolder', type=str, default='../results', help='Folder to output records')
    args = parser.parse_args()
    
    train_config = {
        'lr': args.lr,
        'emb_dim': args.fdim,
        'clip': args.clip,
        'epochs': args.epochs,
        'alpha_dist': 0.5,
        'encoder': '1-gnn' if args.estimator == '1GNN_HSIC' else 'gcn'
    }
    
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
    for seed in tqdm(range(seed_start, seed_start+seeds)):
        for net_type in net_types:
            for N in Ns:
                for m,p in zip(ms, ps):
                        set_seed(seed)
                        sk = NetworkSkeleton(nscm, net_type, seed=seed, N=N, m=m, p=p)
                        for param, values in experiment.items():
                            for value in values:
                                function_config['taus'][param] = value
                                set_seed(seed)
                                functions = FunctionsSetup1(seed=seed, **function_config)
                                sk.instance(functions)
                                A,X,T,Y = load_data(sk)
                                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                                A,X,T,Y = A.to(device), X.to(device), T.to(device), Y.to(device)
                                X = std_scaling(X)
                                model = train_gnn_hsic(train_config, X, A, T, Y)
                                with torch.no_grad():
                                    y_t0, y_t1, _ = model(X, A, T)
                                    y_t1 = y_t1.view(-1)
                                    y_t0 = y_t0.view(-1)
                                err_ate = np.abs(sk.true_effects.mean() - (y_t1.mean() - y_t0.mean()).cpu().numpy())
                                err_pehe = np.sqrt(np.mean((sk.true_effects - (y_t1 - y_t0).cpu().numpy())**2))
                                err_avg = np.sqrt(np.mean((sk.true_effects - (y_t1.mean() - y_t0.mean()).cpu().numpy())**2))
                                record = {
                                    'net_type': net_type, 'N': N, 'm': m, 'p': p, 'seed':seed,
                                    # 'learner': learner, 'adjustment': adjustment,
                                    param: value[0] if type(value)==list else value,
                                    'estimator': f'{args.estimator}',
                                    'err_ate': err_ate,
                                    'err_pehe': err_pehe,
                                    'err_avg': err_avg
                                }
                                print(record)
                                records.append(record)
                        recordDf = pd.DataFrame(records)
                        recordDf.to_csv(out_file, index=False)
                    
    
if __name__=='__main__':
    main()