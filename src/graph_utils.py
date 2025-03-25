import os
import numpy as np
import pandas as pd
import networkx as nx
from graphviz import Source, Digraph
import pydotplus
from networkx.drawing.nx_pydot import to_pydot

# Graph format conversions

def graphviz_to_nx(dag):
    dotplus = pydotplus.graph_from_dot_data(dag.source)
    nx_graph = nx.nx_pydot.from_pydot(dotplus)
    return nx_graph

def nx_to_graphviz(G, filename='anc'):
    dotG = to_pydot(G)
    dotG.set_graph_defaults(**{'nodesep':0.15, 'fontname':"times-bold"})
    dot = dotG.to_string()
    g = Source(dot, filename=filename)
    return g

def dot_file_to_nx(filepath):
    with open(filepath, 'r') as file:
        source = file.readlines()
        source = ''.join(source)
    dotplus = pydotplus.graph_from_dot_data(source)
    dotplus.set_strict(True)
    nx_graph = nx.nx_pydot.from_pydot(dotplus)
    return nx_graph

def dot_file_to_graphviz(filepath):
    dag = Source.from_file(filepath)
    return dag

def adj_to_nx(filepath):
    df = pd.read_csv(filepath, index_col='Unnamed: 0')
    G = nx.from_pandas_adjacency(df, create_using=nx.DiGraph())
    return G

def df_to_nx(df):
    nxDag = nx.DiGraph()
    nxDag.add_edges_from(df.values.tolist())
    return nxDag

def nx2edgesDf(G):
    return pd.DataFrame(list(G.edges()), columns=['from', 'to'])

# Getting parents

def get_parents(G, node):
    parents = [x.strip() for x in G.predecessors(node)]
    return set(parents)

def get_children(G, node):
    children = [x.strip() for x in G.successors(node)]
    return set(children)

def get_ancestors(G, node):
    ancestors = nx.algorithms.dag.ancestors(G, node)
    return ancestors

def get_descendents(G, node):
    descendents = nx.algorithms.dag.descendants(G, node)
    return descendents