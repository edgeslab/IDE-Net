import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from timeit import default_timer as timer

import time
import random
import os
from tqdm import tqdm
import random
import yaml
from datetime import datetime
from run_gnn_tarnet_synthetic import *
from SemiSyntheticData import BlogCatalogData, FlickrData

# os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
# os.environ['RAY_PICKLE_VERBOSE_DEBUG']='1'
# os.environ['TUNE_MAX_PENDING_TRIALS_PG']='8'


def load_data(ssData, seed, **config):
    ssData.reset_params(seed)
    treatments = ssData.treatments()
    outcomes = ssData.outcomes(0, treatments, **config)
    A = torch.FloatTensor(ssData.net.toarray())
    T = treatments
    X = ssData.features
    Y = outcomes
    W = torch.ones_like(ssData.row).view(-1,1)*1.0
    return A,X,T,Y,W

def get_causal_motifs(T, A):
    assignments = T
    data = []
    for i in tqdm(range(len(A))):
        neighbors = A[i].nonzero().view(-1)
        neighbor = len(neighbors)
        bb_1 = np.sum([assignments[j] for j in neighbors])
        bb_0 = neighbor - bb_1
        bbb_0 = 0
        bbb_1 = 0
        bbb_2 = 0
        bbn_0 = 0
        bbn_1 = 0
        bbn_2 = 0
        for j in neighbors:
            for k in neighbors:
                if k > j:
                    if A[j,k]==1:
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

        data.append([bb_0, bb_1, bbb_0, bbb_1, bbb_2, bbn_0, bbn_1, bbn_2])
    return torch.FloatTensor(data)


def main():
    parser = argparse.ArgumentParser(description='Run GNN-based Heterogeneous Network Effects Experiments')
    parser.add_argument('--config', type=str, help='YAML file with config for experiments')
    parser.add_argument('--net', type=str, default='BlogCatalog', choices=['BlogCatalog', 'Flickr'], help='Network for semi-synthetic data')
    parser.add_argument('--exposure', type=str, choices=['hom', 'het', 'both', 'no'], default='both',
                       help='Experiment for homogeneous or heterogeneous exposure') 
    parser.add_argument('--vanilla', type=str, choices=['yes', 'no'], default='no')
    parser.add_argument('--maxiter', type=int, default=300, help='Maximum epochs')
    parser.add_argument('--val', type=float, default=0.2, help='Fraction of nodes used for validation and early stopping')
    parser.add_argument('--lr', type=float, default=0.02, help='Learning rate for encoder')
    parser.add_argument('--lrest', type=float, default=0.2, help='Learning rate for learner')
    parser.add_argument('--lrstep', type=int, default=50, help='Change LR after N steps')
    parser.add_argument('--lrgamma', type=float, default=0.5, help='lr=gamma*lr after N steps')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Regularization')
    parser.add_argument('--clip', type=float, default=3, help='Clip gradient norm')
    parser.add_argument('--max_patience', type=int, default=300, help='Early stopping if loss not improved')
    parser.add_argument('--fdim', type=int, default=64, help='Hidden layer (node embedding) dimension')
    parser.add_argument('--edim', type=int, default=4, help='Hidden layer (edge embedding) dimension')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout')  
    parser.add_argument('--inlayers', type=int, default=2, help='MLP layers for feature/exposure mapping')
    parser.add_argument('--alpha', type=float, default=0., help='Alpha like CFR estimator')
    parser.add_argument('--normY', type=int, default=0, help='Normalize Y')
    parser.add_argument('--isolated', type=int, default=0, help='Evaluate isolated direct effects')
    parser.add_argument('--motifs', type=int, default=0, help='Get causal network motifs')    
    parser.add_argument('--verbose', type=int, default=0, help='Verbose for parameters tuning')
    parser.add_argument('--outfolder', type=str, default='../results', help='Folder to output records')
    
    args = parser.parse_args()
    
    print(f'Exposure type: {args.exposure}, Vanilla: {args.vanilla}')
    
    if args.exposure == 'both':
        exposure_types = [1,2]
    elif args.exposure == 'het':
        exposure_types = [2]
    elif args.exposure == 'hom':
        exposure_types = [1]
    else:
        exposure_types = [0]
    
    if args.vanilla == 'yes':
        vanilla = True
    else:
        vanilla = False
        
    isolated = (args.isolated == 1)
        
    config = None
    exp_name = args.config.split('/')[-1].replace('.yaml', '')
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)
        print(config)   
    seeds = config['seeds']
    seed_start = config.get('seed_start', 0)
    experiment = config['experiment']
    now = datetime.now()
    suffix = now.strftime('%Y-%m-%d-%H-%M')
    out_file = os.path.join(args.outfolder, f'records_{exp_name}_{suffix}.csv')
    records = []
    if args.alpha > 0:
        estimator_name = 'CANE_CFR'
    else:
        estimator_name = 'CANE_TARNet'

    if args.net == 'BlogCatalog':
        ssData = BlogCatalogData()
    elif args.net == 'Flickr':
        ssData = FlickrData()
    else:
        raise Exception("Invalid choice")
    normX, normY = True, args.normY>0
    print(f"========normX: {normX}, normY: {normY}=========")
    for seed in tqdm(range(seed_start, seed_start+seeds)):
                    # set_seed(seed)
                    # sk = NetworkSkeleton(nscm, net_type, seed=seed, N=N, m=m, p=p)
        for param, values in experiment.items():
            for value in values:
                config[param] = value
                set_seed(seed)
                A, X, T, Y, W = load_data(ssData, seed, **config)
                ssData.true_effects = ssData.true_effects.numpy()
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                if args.motifs > 0:
                    motifs = get_causal_motifs(T, A)
                    motifs = motifs.to(device)
                else:
                    motifs = None
                A, X, T, Y, W = A.to(device), X.to(device), T.to(device), Y.to(device), W.to(device)       
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
                                exp_name=exp_name
                                )
                    record = {
                        'net_type': args.net,  'seed':seed,
                        # 'learner': learner, 'adjustment': adjustment,
                        param: value[0] if type(value)==list else value,
                        'estimator': estimator_name,
                        'exposure_type': exposure_type,
                    }
                    for metric in ['loss', 'regloss']:
                        err_ate, err_pehe, err_avg, best_cfg = evaluate(A, X, T, W, Y, ymean, ystd, trials, ssData, metric, normY=normY)
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
                            