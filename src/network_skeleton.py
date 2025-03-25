import numpy as np
import pandas as pd
import networkx as nx
from nscm import Variable
from tqdm import tqdm
import torch

class NSCMGenerator(object):
            
    @staticmethod
    def topological_order(nscm):
        raw_order = nx.topological_sort(nscm.a_graph)
        order = []
        for attr in raw_order:
            if attr not in nscm.latent_attributes() and attr not in nscm.selection_attributes():
                order.append(attr)
        return order
    
    @staticmethod
    def generate(sk, functions):
        """
        sk: skeleton object
        """
        order = NSCMGenerator.topological_order(sk.nscm)
        for attr in order:
            values = getattr(functions, attr)(sk)
            if attr in sk.nscm.entity_attributes():
                sk.nodes_df[attr] = values
            else:
                sk.edges_df[attr] = values
                
class NetworkSkeleton(object):
    """
    Class to encapsulate the network data and access the data with relational variables
    """
    # Types of networks
    # synthetic exogenous
    BARABASI_ALBERT = 'barabasi_albert'
    ERDOS_RENYI = 'erdos_renyi'
    WATTS_STROGATZ = 'watts_strogatz'
    
    # synthetic endogenous, for example homophily
    ENDOGENOUS = 'endogenous'
    # semi-synthetic or realworld networks
    EXOGENOUS = 'exogenous'
    
    SKELETON_TYPES = [BARABASI_ALBERT, ERDOS_RENYI, WATTS_STROGATZ, ENDOGENOUS, EXOGENOUS]
    
    def __init__(self, nscm, skeleton_type, seed=0, **kwargs):
        """
        @params
        nscm: NSCM object
        skeleton_type: Type of skeleton: [BARABASI_ALBERT, ERDOS_RENYI, WATTS_STROGATZ, ENDOGENOUS, EXOGENOUS]
        net: Exogeneous network
        nodes_df: Node attributes for each node
        edges_df: Edge attributes for each edge
        **kwargs:Parameter for skeleton types
        BARABASI_ALBERT: N (number of nodes), m (preferential attachment parameter)
        ERDOS_RENYI: N (number of nodes), p (edge probability)
        ENDOGENOUS: N (number of nodes)
        EXOGENOUS: network, nodes_df, edges_df (edge tuple in ascending order for undirected edges)
        """
        self.nscm = nscm
        self.seed = seed
        self.skeleton_type = skeleton_type
        if self.skeleton_type == NetworkSkeleton.EXOGENOUS:
            self.net = kwargs['network']
            self.nodes_df = kwargs['nodes_df']
            self.edges_df = kwargs['edges_df']
            self.N = len(self.nodes_df)
        else:
            self.net = None
            self.nodes_df = None
            self.edges_df = None
            self.N = None
        self.functions = None 
        self.true_effects = None
        self.generate_network(self.skeleton_type, **kwargs)
        
    
        
    def generate_network(self, skeleton_type, **kwargs):
        self.N = kwargs['N']
        if skeleton_type == NetworkSkeleton.BARABASI_ALBERT:
            m = kwargs['m']
            self.net = nx.barabasi_albert_graph(self.N, m, seed=self.seed)
        elif skeleton_type == NetworkSkeleton.ERDOS_RENYI:
            p = kwargs['p']
            self.net = nx.erdos_renyi_graph(self.N, p=p, seed=self.seed)
        elif skeleton_type == NetworkSkeleton.WATTS_STROGATZ:
            p = kwargs['p']
            self.net = nx.watts_strogatz_graph(self.N, k=int(p*self.N), p=0.6, seed=self.seed)
        else:
            raise NotImplemented
            
        edgesList = list(self.net.edges)
        rel_attrs = (self.nscm.relationship_attributes() - 
                     self.nscm.latent_attributes() - 
                     self.nscm.selection_attributes()
                    )
        ent_attrs = (self.nscm.entity_attributes() - 
                 self.nscm.latent_attributes() - 
                 self.nscm.selection_attributes())
        self.edges_df = pd.DataFrame(np.zeros((len(edgesList), len(rel_attrs)), dtype='float64'),
                                     index=edgesList, columns=rel_attrs)
        # self.edges_df[self.nscm.exist] = 1
        self.nodes_df = pd.DataFrame(np.zeros((self.N, len(ent_attrs)), dtype='float64'),
                                     columns=ent_attrs)

    def instance(self, functions):
        self.functions = functions
        NSCMGenerator.generate(self, functions)
        self.nodes_df = self.nodes_df.astype('float64')
        
    def adj(self):
        return dict(self.net.adjacency())
    
    def am(self, attr, fill=0):
        A = torch.ones((self.N, self.N))*fill
        indices = torch.LongTensor(self.edges_df.index)
        A[indices[:,0], indices[:,1]] = torch.Tensor(self.edges_df[attr].values)
        A[indices[:,1], indices[:,0]] = torch.Tensor(self.edges_df[attr].values)
        return A
    
    def E(self, attr, agg=None):
        return self.nodes_df[attr].values

    def ER(self, attr, agg=False, fill=torch.nan):
        val = self.am(attr, fill=fill)
        if agg:
            return torch.nansum(val, axis=0)#, sk.N - np.count_nonzero(np.isnan(val), axis=0)
        else:
            return val

    def ERE(self, attr, agg=False):
        inds = torch.ones((self.N, self.N)).fill_diagonal_(torch.nan)
        val = torch.Tensor(self.nodes_df[attr].values)
        return val*inds

    def ERER(self, attr, agg=None):
        if agg is None:
            agg = 'degree'
        val = self.am(attr, fill=0)
        val = val.sum(axis=0) - val
        val.fill_diagonal_(torch.nan)
        return val
    
    def ERERxER(self, attr, agg=None):
        if agg is None:
            agg == 'mutual_friends'
        val = self.am(attr, fill=0)
        val = torch.einsum('ij,kj->ik', val, val)
        val.fill_diagonal_(torch.nan)
        return val
    
    def ERxERER(self, attr, agg=None):
        return self.ERERxER(attr, agg)


    def get(self, var, agg=None):
        v = Variable(var)
        val = getattr(self, v.path)(v.attribute, agg=agg)
        return val
        
    
    def E_id(self, id, var, nodes_df):
        return nodes_df.loc[id, var.attribute]

    def ERE_id(self, id, var, nodes_df):
        return nodes_df.loc[(nodes_df.index != id), var.attribute]

    def ER_id(self, id, var, nodes_df, ignore=None):
        N = len(nodes_df)
        # values = pd.Series(np.ones(N)*np.nan, index=nodes_df.index)
        peers = [*self.adj()[id]]
        if ignore is not None:
            if ignore in self.adj()[id]:
                peers.remove(ignore)
        values = pd.Series(np.ones(len(peers))*np.nan, index=peers)
        if var.attribute == self.nscm.exist:
            values[peers] = 1
        else:
            edges = list(map(lambda other: (id, other) if id<other else (other, id), peers))
            values[peers] = self.edges_df.loc[edges, var.attribute]
        return values[values.index != id]

    def ERER_id(self, id, var, nodes_df):
        others = nodes_df.loc[(nodes_df.index != id), :]
        values = dict(map(lambda other_id: (other_id, self.ER(other_id, var, others, ignore=id)),
                          others.index))
        return values
    
    def get_id(self, var, id):
        if type(var) == str:
            var = Variable(var)
        return getattr(self, f'{var.path}_id')(id=id, var=var, nodes_df=self.nodes_df)
    
    def weighted_mean(self, var, weight):
        deno = weight.sum()
        # print(var,weight)
        # assert deno != 0
        return (var*weight).sum() / deno
    
    # TODO optimize
    def similarity(self, var1, var2, gamma=0.2):
        return np.exp(-gamma*np.square(var1-var2))
    
    def degree_agg(self, Er1):
        """
        Er1 size = (N-1 x (num_friends except i)) i.e. ERER.Er
        """
        return pd.Series(dict(map(lambda key: (key, Er1[key].sum()), Er1.keys())))

    def value_agg(self, Zr1):
        """
        ERER.Zr = sum(ERER.Zr)/count(ERER.Zr) # avg weight among friends
        """
        return pd.Series(dict(map(lambda key: (key,Zr1[key].sum()/len(Zr1[key])), Zr1.keys())))

    def mutual_friends_agg(self, Er, Er1):
        Er = set(Er.index)
        return pd.Series(dict(map(lambda key: (key,
                                               len(Er.intersection(set(Er1[key].index)))),
                                  Er1.keys())))
    def hops_agg(self, Er, Er1, nhop=None):
        result = pd.Series(0,index=Er1.keys())
        result[Er.index] = 1
        if nhop is None:
            nhop = len(result)
        hop = 1
        explored = set(Er.index)
        while hop < nhop:
            for ind in result[result==hop].index:
                children = set(Er1[ind].index) - explored
                result[children] = hop + 1
                explored = explored.union(children)
            hop += 1
        return result
