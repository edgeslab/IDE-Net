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

from nscm import *
from network_skeleton import NetworkSkeleton, NSCMGenerator
from nscm_functions import FunctionsSetup1

from model import NetEsimator
from experiment_minimal import ExperimentMinimal
from baselineModels import GCN_DECONF,CFR,GCN_DECONF_INTERFERENCE,CFR_INTERFERENCE
from tnet.targetedModel_DoubleBSpline import TargetedModel_DoubleBSpline


def append_to_row(series, data_dict):
    row = series.to_dict()
    row.update(data_dict)
    return pd.Series(row)

def set_seed(seed=1):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def load_data(sk):
    A = torch.FloatTensor(sk.am('Er'))
    T = torch.FloatTensor(sk.nodes_df['X'].values)
    Y = torch.FloatTensor(sk.nodes_df['Y'].values)
    features = sorted(set(sk.nodes_df.columns) - set(['X', 'Y']))
    X = torch.FloatTensor(sk.nodes_df[features].values)
    return A, X, T, Y

def estimator_args():
    args = {
        'cuda': 1,
        'seed': 24,
        # 'dataset': 'BC',
        # 'expID': 4,
        # 'flipRate': 1,
        'alpha': 0.5,
        'gamma': 0.5,
        'epochs': 100,
        'lr': 1e-3,
        'lrD': 1e-3,
        'lrD_z': 1e-3,
        'weight_decay': 1e-5,
        'dstep': 1,
        'd_zstep': 1,
        'pstep': 1,
        'normy': 1, 
        'hidden': 32,
        'dropout': 0.1, 
        'save_intermediate': 0,
        'model': 'NetEsimator',
        'alpha_base': 0.5,
        'printDisc': 0,
        'printDisc_z': 0,
        'printPred': 0,
        'lr_1step': 1e-4,
        'lr_2step': 1e-2,
        'weight_decay_tr': 1e-3,
        'alpha_tr': 0.5,
        'gamma_tr': 1.,
        'beta_tr': 20.,
        'pre_train_step_tr':0,
        'iter_2step_tr': 50,
        'loss_2step_with_ly':0,
        'loss_2step_with_ltz':0,

    }
    args['cuda'] = args['cuda'] and torch.cuda.is_available()
    return args

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

        data.append([bb_0, bb_1, bbb_0, bbb_1, bbb_2, bbn_0, bbn_1, bbn_2])
    return torch.FloatTensor(data)
    
def main():
    parser = argparse.ArgumentParser(description='Run Heterogeneous Network Effects Experiments')
    parser.add_argument('--config', type=str, help='YAML file with config for experiments')
    parser.add_argument('--estimator', type=str, default='TARNet_INTERFERENCE',
                        choices=['TNet', 'NetEstimator', 'ND', 'TARNet', 'CFR', 'CFR_INTERFERENCE', 
                                 'ND_INTERFERENCE', 'TARNet_INTERFERENCE', 'TARNet_MOTIFS', 'ND_MOTIFS'],
                        help='Estimator to estimate heterogeneous effects')
    parser.add_argument('--num_grid', type=int, default=20, help='Number of grids for piecewise regression of Z in TNet.')  # 10000
    parser.add_argument('--tr_knots', type=float, default=0.1, help='trade-off of targeted regur in TargetedModel')

    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--isolated', type=int, default=0, help='Isolated direct effects')
    parser.add_argument('--epochs', type=int, default=300, help='Epochs')
    parser.add_argument('--outfolder', type=str, default='../results', help='Folder to output records')
    args = parser.parse_args()
    config = None
    exp_name = args.config.split('/')[-1].replace('.yaml', '')
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)
        print(config)
    model = config['model']
    nscm = NSCM(model)
    print(nscm.model)
    # nagg = NAGG(nscm)
    # X = config['X']
    # Y = config['Y']
    # W,Z = nagg.cane_adjustment_set(X, Y, nagg.G,
    #                    S=nscm.selection_attributes(),
    #                    L=nscm.latent_attributes(),
    #                    include=['ER.Er']) # If valid include relationship existence in the minimal adjustment set
    # Z = sorted(Z, key=lambda x: len(Variable(x).path)) # CANE Adjustment set, only potential sources of heterogeneity
    # W = sorted(W, key=lambda x: len(Variable(x).path)) # Minimal adjustment set + network features
    # print(f'Minimal: {W}\nCANE: {Z}')
    
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
    for net_type in net_types:
        for N in Ns:
            for m,p in zip(ms, ps):
                for seed in tqdm(range(seed_start, seed_start+seeds)):
                    set_seed(seed)
                    sk = NetworkSkeleton(nscm, net_type, seed=seed, N=N, m=m, p=p)
                    for param, values in experiment.items():
                        for value in values:
                            function_config['taus'][param] = value
                            set_seed(seed)
                            functions = FunctionsSetup1(seed=seed, **function_config)
                            sk.instance(functions)
                            trainA,trainX,trainT,POTrain = load_data(sk)
                            est_args = estimator_args()
                            est_args['epochs'] = args.epochs
                            est_args['lr'] = args.lr
                            est_args['isolated'] = (args.isolated == 1)
                            if est_args['cuda']:
                                trainA,trainX,trainT,POTrain = trainA.cuda(), trainX.cuda(), trainT.cuda(), POTrain.cuda()
                            est_args['model'] = args.estimator
                            motifs = None
                            if est_args['model'] == 'TNet':
                                model = TargetedModel_DoubleBSpline(Xshape=trainX.shape[1], hidden=est_args['hidden'], dropout=est_args['dropout'], num_grid=args.num_grid, tr_knots=args.tr_knots)

                            elif est_args['model'] == "NetEstimator":
                                model = NetEsimator(Xshape=trainX.shape[1],hidden=est_args['hidden'],dropout=est_args['dropout'])
                            elif est_args['model'] == "ND":
                                model = GCN_DECONF(nfeat=trainX.shape[1], nhid=est_args['hidden'],dropout=est_args['dropout'])
                            elif est_args['model'] == "TARNet":
                                model = GCN_DECONF(nfeat=trainX.shape[1], nhid=est_args['hidden'],dropout=est_args['dropout'])
                            elif est_args['model'] == "CFR":
                                model = CFR(nfeat=trainX.shape[1], nhid=est_args['hidden'],dropout=est_args['dropout'])

                            elif est_args['model'] == "CFR_INTERFERENCE":
                                model = CFR_INTERFERENCE(nfeat=trainX.shape[1], nhid=est_args['hidden'],dropout=est_args['dropout'])
                            elif est_args['model'] == "ND_INTERFERENCE":
                                model = GCN_DECONF_INTERFERENCE(nfeat=trainX.shape[1], nhid=est_args['hidden'],dropout=est_args['dropout'])
                            elif est_args['model'] == "TARNet_INTERFERENCE":
                                model = GCN_DECONF_INTERFERENCE(nfeat=trainX.shape[1], nhid=est_args['hidden'],dropout=est_args['dropout'])
                            elif est_args['model'] == "TARNet_MOTIFS" or est_args['model'] == "ND_MOTIFS":
                                motifs = get_causal_motifs(sk)
                                motifs = motifs.to(trainA.device)
                                est_args['motifs'] = motifs
                                model = GCN_DECONF_INTERFERENCE(nfeat=trainX.shape[1], nhid=est_args['hidden'],dropout=est_args['dropout'], num_motifs=motifs.shape[1])
                            exp = ExperimentMinimal(est_args,model,trainA, trainX, trainT,POTrain)
                            exp.train()
                            y_t1, y_t0 = exp.compute_individual_effect(trainA, trainX, trainT)
                            err_ate = np.abs(sk.true_effects.mean() - (y_t1.mean() - y_t0.mean()).cpu().numpy())
                            err_pehe = np.sqrt(np.mean((sk.true_effects - (y_t1 - y_t0).cpu().numpy())**2))
                            err_avg = np.sqrt(np.mean((sk.true_effects - (y_t1.mean() - y_t0.mean()).cpu().numpy())**2))
                            record = {
                                'net_type': net_type, 'N': N, 'm': m, 'p': p, 'seed':seed,
                                # 'learner': learner, 'adjustment': adjustment,
                                param: value[0] if type(value)==list else value,
                                'estimator': args.estimator,
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
