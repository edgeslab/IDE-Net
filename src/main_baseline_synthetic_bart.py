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
from torch_sparse import SparseTensor
# https://github.com/JakeColtman/bartpy
from bartpy.sklearnmodel import SklearnModel

from nscm import *
from network_skeleton import NetworkSkeleton, NSCMGenerator
from nscm_functions import FunctionsSetup1

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


def agg(trainA, trainX, trainT):
    edge_index = trainA.nonzero().t()
    adj_t = SparseTensor(row=edge_index[0], col=edge_index[1]).t()
    deg = adj_t.sum(1).view(-1,1).clip(1)
    aggX = adj_t.matmul(trainX)/deg
    aggT = adj_t.matmul(trainT.view(-1,1))/deg
    return deg, aggX, aggT


class BART():
    def __init__(self):
        self.model = SklearnModel(n_jobs=-1, store_in_sample_predictions=False)  # Use default parameters
        self.isolated = False
    
    def fit(self, trainA, trainX, trainT, trainY):
        dataX = torch.cat([trainT.view(-1,1), trainX], dim=1)
        self.model.fit(dataX.numpy(), trainY.numpy())
    
    def compute_individual_effect(self, trainA, trainX, trainT):
        dataX_t1 = torch.cat([torch.ones_like(trainT).view(-1,1), trainX], dim=1)
        dataX_t0 = torch.cat([torch.zeros_like(trainT).view(-1,1), trainX], dim=1)
        y_t1 = self.model.predict(dataX_t1.numpy())
        y_t0 = self.model.predict(dataX_t0.numpy())
        return y_t1, y_t0
    
class BART_NET():
    def __init__(self):
        self.model = SklearnModel(n_jobs=-1, store_in_sample_predictions=False)  # Use default parameters
        self.isolated = False
    
    def fit(self, trainA, trainX, trainT, trainY):
        deg, aggX, aggT = agg(trainA, trainX, trainT)
        dataX = torch.cat([trainT.view(-1,1), trainX, aggX, deg], dim=1)
        self.model.fit(dataX.numpy(), trainY.numpy())
    
    def compute_individual_effect(self, trainA, trainX, trainT):
        deg, aggX, aggT = agg(trainA, trainX, trainT)
        dataX = torch.cat([trainX, aggX, deg], dim=1)
        dataX_t1 = torch.cat([torch.ones_like(trainT).view(-1,1), dataX], dim=1)
        dataX_t0 = torch.cat([torch.zeros_like(trainT).view(-1,1), dataX], dim=1)
        y_t1 = self.model.predict(dataX_t1.numpy())
        y_t0 = self.model.predict(dataX_t0.numpy())
        return y_t1, y_t0

class BART_INT():
    def __init__(self, isolated=False):
        self.model = SklearnModel(n_jobs=-1, store_in_sample_predictions=False)  # Use default parameters
        self.isolated = isolated
    
    def fit(self, trainA, trainX, trainT, trainY):
        deg, aggX, aggT = agg(trainA, trainX, trainT)
        dataX = torch.cat([trainT.view(-1,1), trainX, aggX, deg, aggT], dim=1)
        self.model.fit(dataX.numpy(), trainY.numpy())
    
    def compute_individual_effect(self, trainA, trainX, trainT):
        deg, aggX, aggT = agg(trainA, trainX, trainT)
        dataX = torch.cat([trainX, aggX, deg], dim=1)
        if self.isolated:
            aggT = torch.zeros_like(aggT)
        dataX_t1 = torch.cat([torch.ones_like(trainT).view(-1,1), dataX, aggT], dim=1)
        dataX_t0 = torch.cat([torch.zeros_like(trainT).view(-1,1), dataX, aggT], dim=1)
        y_t1 = self.model.predict(dataX_t1.numpy())
        y_t0 = self.model.predict(dataX_t0.numpy())
        return y_t1, y_t0

def main():
    parser = argparse.ArgumentParser(description='Run Heterogeneous Network Effects Experiments for BART')
    parser.add_argument('--config', type=str, help='YAML file with config for experiments')
    parser.add_argument('--estimator', type=str, default='BART_NET',
                        choices=['BART_NET', 'BART_INT', 'BART'],
                        help='Estimator to estimate heterogeneous effects with BART')
    parser.add_argument('--isolated', type=int, default=0, help='Isolated direct effects')
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

                            if args.estimator == "BART":
                                model = BART()
                            elif args.estimator == "BART_NET":
                                model = BART_NET()
                            elif args.estimator == "BART_INT":
                                model = BART_INT(isolated=args.isolated==1)
                            else:
                                raise Exception("Invalid choice")

                            model.fit(trainA, trainX, trainT, POTrain)
                            y_t1, y_t0 = model.compute_individual_effect(trainA, trainX, trainT)
                            err_ate = np.abs(sk.true_effects.mean() - (y_t1.mean() - y_t0.mean()))
                            err_pehe = np.sqrt(np.mean((sk.true_effects - (y_t1 - y_t0))**2))
                            err_avg = np.sqrt(np.mean((sk.true_effects - (y_t1.mean() - y_t0.mean()))**2))
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
            
            
