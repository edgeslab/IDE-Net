import numpy as np
import pandas as pd
import torch
import networkx as nx
def wsum(X, weight):
    return torch.nan_to_num(torch.div(torch.einsum('ij,i->j', weight.nan_to_num(), X), torch.nansum(weight, axis=1)+1e-8))

def softmax(X, weight, mask):
    weight = weight.nan_to_num()
    weight = weight - weight.max()
    weight = weight
    weight = weight.exp()*mask.nan_to_num()
    deno = weight.sum(1, keepdim=True)
    weight = weight.div(deno+1e-12)*X.view(1,-1)
    return weight.sum(1)

def gaussian_kernel(X, Y=None, gamma=None):
    """
    X: (N,d)
    Y: (M,d)
    gamma: 1/d default value
    Returns:
    (N,M)
    """
    N,d = X.shape
    if gamma is None:
        gamma = 1/d
    if Y is None:
        Y = X
    x_i = X[:, None, :]  # (N, 1, d)
    y_j = Y[None, :, :]  # (1, M, d)
    D_ij = ((x_i - y_j) ** 2).sum(-1)  # (N, M) symbolic matrix of squared distances
    return (-gamma * D_ij).exp()  # (N, M) symbolic Gaussian kernel matrix

class FunctionsSetup1(object):
    def __init__(self, seed=0, **kwargs):
        """
        @params
        kwargs: Dictionary with the following keys
        
        alphas : Dirichelet alphas to generate P(Z=z) with 3 levels. Eg: [5,3,3]
        betas: Beta distribution params [a,b] to generate C in between 0 to 1
        gamma: RBF Kernel similarity measure control parameter: Eg: 0.5
        
        taus: {
            'direct': -2, 
            'attr_em': -1,
            'net_em': 1,
            'confounder': -3,
            'exposure': -0,
            'het_exposure': -0,
            'treatment_influence': 0.3
        }, where taus are effect sizes
        
        """
        self.seed = seed
        np.random.seed(self.seed)
        self.kwargs = kwargs
        self.alphas = self.kwargs.get("alphas", [5,4,4])
        self.betas = self.kwargs.get("betas", [0.6,0.6])
        self.gamma = self.kwargs.get("gamma", 0.5)
        self.taus = self.kwargs.get('taus', dict())
        variableMap = {
            'direct': ('tau_X', [-2]),
            'attr_em': ('tau_Z', [1]),
            'confounder': ('tau_C', [-3]),
            'net_em': ('tau_Er', [-1]),
            'hom_exposure': ('tau_X1_hom', [0]),
            'het_exposure': ('tau_X1_het_tie', [0]), 
            'het_exposure_mut_frns': ('tau_X1_het_conn', [0]),
            'het_exposure_sim': ('tau_X1_het_sim', [0]),
            'het_exposure_deg': ('tau_X1_deg', [0]),
            'exposure': ('tau_X1', [0]),
            'treatment_influence': ('tau_C1', [0.3]),
            'het_exposure_div': ('tau_X1_div', [0]),
            'exposure_em': ('tau_X1_em', [0]),
            }
        for key,var in variableMap.items():
            value = self.taus.get(key, var[1])
            value = np.random.choice(value)
            print(var[0], value)
            setattr(self, var[0], value)

        self.activation = self.kwargs.get('activation', 'linear')
        
    
    # Er implicit edge attribute
    def Er(self, sk):
        return np.ones(len(sk.edges_df), dtype='float64')
    
    # Zr edge weights like friendship duration
    def edge_weight(self, df):
        np.random.seed(self.seed)
        # betas = np.random.uniform(0.7, 1, size=2)
        # zr = np.random.beta(*betas, size=len(df))
        # zr = 1 + (3 - 1)*zr #scaling
        zr = np.random.uniform(1, 10, size=len(df))
        return zr

    def Zr(self, sk):
        return sk.edges_df.groupby(by=lambda x: x[0])['Zr'].transform(lambda df: self.edge_weight(df))
    
    # Z like Age. Biased toward younger population
    def Z(self, sk):
        np.random.seed(self.seed)
        values = np.random.choice(range(len(self.alphas)), size=sk.N, p=np.random.dirichlet(self.alphas))
        return values
    
    # C like Disposition. Beta distribution to give values between 0 and 1, open to reserved
    def C(self, sk):
        np.random.seed(self.seed)
        values = np.random.beta(*self.betas, size=sk.N)
        return values
    
     # X like privacy settings. On if reserved disposition, maybe influenced by peers
    def X(self, sk):
        np.random.seed(self.seed)
        np.random.seed(self.seed)
        C = torch.Tensor(sk.nodes_df['C'].values)
        Er = torch.Tensor(sk.am('Er', fill=torch.nan))
        values = self.tau_C1*(torch.divide(torch.nansum(C*Er,dim=1), torch.nansum(Er, dim=1))) + (1-self.tau_C1)*C
        na = torch.isnan(values)
        values[na] = C[na]
        values = torch.bernoulli(values)
        # values = torch.where(values > 0.5, 1.0, 0.0)
        return values  
    
    def get_structural_diversity_treatment(self, sk):
        assignments = sk.nodes_df['X']
        G = sk.net
        structural_diversity = []
        for uid in list(G.nodes):
            structural_diversity.append(
                nx.number_connected_components(nx.subgraph(G, [j for j in nx.neighbors(G, uid) if assignments[j] == 1]))
            )
        return torch.FloatTensor(structural_diversity)
   
    def Y(self, sk):
        np.random.seed(self.seed)
        Er = torch.Tensor(sk.am('Er'))
        Zr = torch.Tensor(sk.am('Zr'))
        X = torch.Tensor(sk.nodes_df['X'].values)
        Z = torch.Tensor(sk.nodes_df['Z'].values)
        C = torch.Tensor(sk.nodes_df['C'].values)

        y = 20.0
        y += self.tau_X*X # direct effect
        y += self.tau_Z*X*Z # attribute effect modifier
        y += self.tau_C*C # confounder
        y += self.tau_C1*self.tau_C*(torch.divide(torch.nansum(C*Er,dim=1), torch.nansum(Er, dim=1))) # confounder

        deg_Er = Er.sum(axis=0)
        inds = torch.ones_like(Er)
        inds.fill_diagonal_(torch.nan)
        deg_Er1 = (Er.sum(axis=0) - Er)*inds

        deg_Er1_75p = torch.nanquantile(deg_Er1, 0.75, dim=1)
        y += self.tau_Er*X*(deg_Er > deg_Er1_75p) # network effect modifier
        
        weight_sim = 0.
        weight_het = 0.
        weight_conn = 0.
        weight_hom = 0.
        weight_deg = 0.
        if self.tau_X1_het_sim != 0:
            weight_sim = Er*gaussian_kernel(Z.reshape(-1,1), gamma=self.gamma)*inds
            weight_sim = self.tau_X1_het_sim*weight_sim
        if self.tau_X1_het_tie != 0:
            weight_het = Zr.pow(2)
            weight_het = self.tau_X1_het_tie*weight_het
        if self.tau_X1_het_conn != 0:
            mut_frns_Er1 = torch.einsum('ij,kj->ik', Er, Er)*inds
            weight_conn = Er*mut_frns_Er1.pow(0.5)
            weight_conn = self.tau_X1_het_conn*weight_conn
        if self.tau_X1_hom !=0:
            weight_hom = self.tau_X1_hom*Er
        if self.tau_X1_deg !=0:
            deg = Er.sum(dim=-1)
            weight_deg = self.tau_X1_deg*(Er*deg)
            
        # if self.tau_X1 !=0:
        weight = weight_sim + weight_het + weight_conn + weight_hom + weight_deg
        # else:
            # weight = Er
        weight = 0. if type(weight) == float else wsum(X, weight)
        
        y += self.tau_X1*weight

                
        if self.tau_X1_div!=0:
            div = self.get_structural_diversity_treatment(sk)
            y += self.tau_X1 * div
        else:
            div = 0.
            
        if self.tau_X1_em!=0:
            y += self.tau_X1_em * X*div
            y += self.tau_X1_em * X*weight
        
        # y += torch.FloatTensor(np.random.uniform(-1,+1, size=sk.N))
        y += torch.FloatTensor(np.random.normal(size=sk.N))

        if self.activation == 'linear':
            true_effects = self.tau_X + self.tau_Z*Z + self.tau_Er * (deg_Er > deg_Er1_75p) + self.tau_X1_em * div + self.tau_X1_em * weight
            sk.true_effects = pd.Series(true_effects, index=sk.nodes_df.index)
        else:
            y = y-19
            y = y.sigmoid()
            true_effects = y * (1-y) * (self.tau_X + self.tau_Z*Z + self.tau_Er * (deg_Er > deg_Er1_75p) + self.tau_X1_em * div + self.tau_X1_em * weight)
            sk.true_effects = pd.Series(true_effects, index=sk.nodes_df.index)
        return y
        