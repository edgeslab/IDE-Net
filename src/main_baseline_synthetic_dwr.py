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
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
import torch_sparse
from torch_sparse import SparseTensor
from torcheval.metrics import R2Score

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

class DWRModel(torch.nn.Module):
    def __init__(self, n_in, n_out=1, n_hemb=32, n_hgcn=32):
        super(DWRModel, self).__init__()
        torch.manual_seed(9)
        self.emb = Linear(n_in, n_hemb)
        self.gcn = GCNConv(n_hemb, n_hgcn)
        # self.out = Linear(n_hgcn+2, n_out)
        self.out = torch.nn.Sequential(Linear(n_hgcn+2, n_hemb), torch.nn.ReLU(), Linear(n_hemb, n_out))
        self.activation = torch.nn.ReLU()
        
    def forward(self, x, A, t, int_t=None, int_z=None):
        h_emb = self.activation(self.emb(x))
        row, col = A.nonzero().t()
        adj_t = SparseTensor(row=row, col=col).t()
        # attention weights
        att_wt = torch.einsum('mh,mh->m', h_emb[row], h_emb[col])
        att_wt = att_wt - att_wt.max()
        att_wt = att_wt.exp()
        deno = adj_t.set_value(att_wt).sum(1) + 1e-8
        att_wt = att_wt.div(deno[row])
        # peer exposure
        exp_t = att_wt*t[col]
        z = adj_t.set_value(exp_t).sum(1)
        # gcn
        rep = h_emb + self.activation(self.gcn(h_emb, adj_t, att_wt))
        if int_t is not None:
            t = int_t
        if int_z is not None:
            z = int_z
        features = torch.cat([rep, t.view(-1,1), z.view(-1,1)], dim=1)
        out = self.out(features)
        return out, rep, z
    
class SampleWeightModel(torch.nn.Module):
    def __init__(self, n_hgcn):
        super(SampleWeightModel, self).__init__()
        torch.manual_seed(9)
        self.wt = Linear(n_hgcn+2, 1)
    
    def forward(self, r, t, z):
        features = torch.cat([r, t.view(-1,1), z.view(-1,1)], dim=1)
        return self.wt(features)
        
        
def train(config, X, A, T, Y):

    lr = config.get('lr', 0.01)
    l2_lambda = config.get('l2_lambda', 1e-5)
    emb_dim = config.get('emb_dim', 32)
    rep_dim = config.get('rep_dim', emb_dim)
    out_dim = config.get('out_dim', 1)
    N, in_dim = X.shape
    device = X.device
    dwr_model = DWRModel(in_dim, out_dim, n_hemb=emb_dim, n_hgcn=rep_dim).to(device)
    wt_model = SampleWeightModel(rep_dim).to(device)
    clip = config.get('clip', 0)
    if  clip > 0:
        modules = [dwr_model, wt_model]
        for module in modules:
            for p in module.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, -clip, clip))

    opt_dwr = torch.optim.Adam(dwr_model.parameters(),
                             lr=lr, 
                             weight_decay=l2_lambda)
    opt_wt = torch.optim.Adam(wt_model.parameters(),
                             lr=lr, 
                             weight_decay=l2_lambda)

    criterion = torch.nn.MSELoss(reduction='none')  # Define loss criterion.
    criterion_cal = torch.nn.BCEWithLogitsLoss()
    r2score = R2Score()
    train_index, val_index = train_test_split(np.arange(X.shape[0]), 
                                          test_size=config.get('num_val', 0.2),
                                          stratify=T.cpu().numpy(),
                                          random_state=9)
    data = Data(x=X,y=Y, train_index=train_index, val_index=val_index).to(A.device)
    best_loss = float('inf')

    epochs = config.get('epochs', 100)
    cal_epochs = config.get('cal_epochs', 5)

    for epoch in range(epochs):
        dwr_model.train()
        opt_dwr.zero_grad()
        y_pred, rep, z = dwr_model(X, A, T)

        T_cal = T[torch.randperm(T.shape[0])]
        z_cal = z[torch.randperm(z.shape[0])]

        rep_cal = torch.cat([rep, rep])
        T_cal = torch.cat([T, T_cal])
        z_cal = torch.cat([z, z_cal])
        Y_cal = torch.cat([torch.ones_like(T), torch.zeros_like(T)])
        perm = torch.randperm(2*T.shape[0])
        rep_cal, T_cal, z_cal, Y_cal = rep_cal[perm].detach(), T_cal[perm], z_cal[perm].detach(), Y_cal[perm]
        
        for cal_epoch in range(cal_epochs):
            wt_model.train()
            opt_wt.zero_grad()
            out_cal = wt_model(rep_cal, T_cal, z_cal)
            # print(out_cal.shape)
            loss_cal = criterion_cal(out_cal, Y_cal.view(-1,1))
            loss_cal.backward()
            opt_wt.step()

        prob_sample = torch.sigmoid(wt_model(rep, T, z))
        pred_loss = criterion(y_pred[data.train_index], Y[data.train_index].view(-1,1))
        if cal_epochs > 0:
            sample_wt = (1 - prob_sample)/prob_sample
            sample_wt = sample_wt.detach()
        else:
            sample_wt = torch.ones_like(T)
        pred_loss = (pred_loss*sample_wt[train_index]).mean()
        pred_loss.backward()
        opt_dwr.step()

        with torch.no_grad():
            dwr_model.eval()
            y_pred, rep, z = dwr_model(X, A, T)
            loss_val = criterion(y_pred[data.val_index], Y[data.val_index].view(-1,1)).mean()
            r2_val = r2score.update(y_pred[data.val_index].view(-1,1),
                                Y[data.val_index].view(-1,1)).compute()
            if epoch % config.get('eval_interval', 300) == 0:
                print(f"Epoch:{epoch}, train_loss:{pred_loss.item()}, val_loss:{loss_val.item()}, val_r2:{r2_val.item()}")
    return dwr_model

def main():
    parser = argparse.ArgumentParser(description='Baseline DWR implementation')
    parser.add_argument('--config', type=str, help='YAML file with config for experiments')
    parser.add_argument('--estimator', type=str, default='DWR', help='Name of the estimator')
    parser.add_argument('--lr', type=float, default=0.2, help="Learning rate")
    parser.add_argument('--fdim', type=int, default=32, help='Hidden layer (node embedding) dimension')
    parser.add_argument('--clip', type=float, default=3, help='Clip gradient')
    parser.add_argument('--epochs', type=int, default=300, help='Number of Epochs')
    parser.add_argument('--calepochs', type=int, default=5, help='Number of calibration Epochs')
    parser.add_argument('--outfolder', type=str, default='../results', help='Folder to output records')
    args = parser.parse_args()
    
    train_config = {
        'lr': args.lr,
        'emb_dim': args.fdim,
        'clip': args.clip,
        'epochs': args.epochs,
        'cal_epochs': args.calepochs
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
    out_file = os.path.join(args.outfolder, f'records_{args.estimator}_{args.calepochs}_{exp_name}_{suffix}.csv')
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
                                dwr_model = train(train_config, X, A, T, Y)
                                t1 = torch.ones_like(T)
                                t0 = torch.zeros_like(T)
                                with torch.no_grad():
                                    y_t1, rep1, z1 = dwr_model(X, A, T, int_t=t1)
                                    y_t0, rep0, z0 = dwr_model(X, A, T, int_t=t0)
                                    y_t1 = y_t1.view(-1)
                                    y_t0 = y_t0.view(-1)
                                err_ate = np.abs(sk.true_effects.mean() - (y_t1.mean() - y_t0.mean()).cpu().numpy())
                                err_pehe = np.sqrt(np.mean((sk.true_effects - (y_t1 - y_t0).cpu().numpy())**2))
                                err_avg = np.sqrt(np.mean((sk.true_effects - (y_t1.mean() - y_t0.mean()).cpu().numpy())**2))
                                record = {
                                    'net_type': net_type, 'N': N, 'm': m, 'p': p, 'seed':seed,
                                    # 'learner': learner, 'adjustment': adjustment,
                                    param: value[0] if type(value)==list else value,
                                    'estimator': f'{args.estimator}-{args.calepochs}',
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
                            