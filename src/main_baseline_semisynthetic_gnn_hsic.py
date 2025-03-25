def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import argparse
import pandas as pd
import numpy as np
import os
import networkx as nx
from tqdm import tqdm
import random
import yaml
from datetime import datetime

import torch
import torch.nn.functional as F

from main_baseline_synthetic_gnn_hsic import train_gnn_hsic

def set_seed(seed=1):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


from SemiSyntheticData import BlogCatalogData, FlickrData
def load_data(ssData, seed, **config):
    ssData.reset_params(seed)
    treatments = ssData.treatments()
    outcomes = ssData.outcomes(0, treatments, **config)
    A = torch.FloatTensor(ssData.net.toarray())
    T = treatments
    X = ssData.features
    Y = outcomes
    W = None
    return A,X,T,Y

def main():
    parser = argparse.ArgumentParser(description='Run Heterogeneous Network Effects Experiments')
    parser.add_argument('--config', type=str, help='YAML file with config for experiments')
    parser.add_argument('--net', type=str, default='BlogCatalog', choices=['BlogCatalog', 'Flickr'], help='Network for semi-synthetic data')
    parser.add_argument('--estimator', type=str, default='1GNN_HSIC', choices=['1GNN_HSIC', 'GCN_HSIC'], help='Name of the estimator')
    parser.add_argument('--lr', type=float, default=0.1, help="Learning rate")
    parser.add_argument('--fdim', type=int, default=64, help='Hidden layer (node embedding) dimension')
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
    
    seeds = config['seeds']
    seed_start = config.get('seed_start', 0)
    # net_types = config['network']['skeleton_type']
    # Ns = config['network']['N']
    # ms = config['network']['m']
    # ps = config['network']['p']
    # function_config = config['functions']
    experiment = config['experiment']
    now = datetime.now()
    suffix = now.strftime('%Y-%m-%d-%H-%M')
    out_file = os.path.join(args.outfolder, f'records_{args.estimator}_{exp_name}_{suffix}.csv')
    records = []
    if args.net == 'BlogCatalog':
        ssData = BlogCatalogData()
    elif args.net == 'Flickr':
        ssData = FlickrData()
    else:
        raise Exception("Invalid choice")
    
    for seed in tqdm(range(seed_start, seed_start+seeds)):
        set_seed(seed)
        for param, values in experiment.items():
            for value in values:
                config[param] = value
                set_seed(seed)
                A,X,T,Y = load_data(ssData, seed, **config)
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                A,X,T,Y = A.to(device), X.to(device), T.to(device), Y.to(device)
                model = train_gnn_hsic(train_config, X, A, T, Y)
                with torch.no_grad():
                    y_t0, y_t1, _ = model(X, A, T)
                    y_t1 = y_t1.view(-1)
                    y_t0 = y_t0.view(-1)
                err_ate = np.abs(ssData.true_effects.numpy().mean() - (y_t1.mean() - y_t0.mean()).cpu().numpy())
                err_pehe = np.sqrt(np.mean((ssData.true_effects.numpy() - (y_t1 - y_t0).cpu().numpy())**2))
                err_avg = np.sqrt(np.mean((ssData.true_effects.numpy() - (y_t1.mean() - y_t0.mean()).cpu().numpy())**2)) 
                
                record = {
                    'net_type':args.net, 'seed':seed,
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
                
