import networkx as nx
import pandas as pd
import numpy as np
import os
from graph_utils import *
import json
import yaml

class Variable(object):
    def __init__(self, var):
        self.var = var
        self.path, self.attribute = var.split(".")
        self.base = self.path[0]
        self.terminal = self.path[-1]

    def __repr__(self):
        return f'{self.path}.{self.attribute}'
    

class Dependency(object):
    def __init__(self, dep):
        cause, effect = dep.split('->')
        self.cause = Variable(cause)
        self.effect = Variable(effect)
    
    def __repr__(self):
        return f'{self.cause}->{self.effect}'
    
class NSCM(object):
    """
    Network structural causal model class
    """
    def __init__(self,model):
        """
        model: JSON model_path or dict
        """
        if type(model)==str:
            if model.endswith(".json"):
                with open(model, 'r') as model_path:
                    self.model = json.load(model_path)
            elif model.endswith(".yaml"):
                with open(model, 'r') as model_path:
                    self.model = yaml.safe_load(model_path)
            else:
                raise Exception("Invalid model format")
        elif type(model)==dict:
            self.model = model
        self.model['Attributes']['Entity'] = set(self.entity_attributes())
        self.exist = self.relationship_attributes()[0]
        self.model['Attributes']['Relationship'] = set(self.relationship_attributes())
        self.model['Attributes']['Latent'] = set(self.latent_attributes())
        self.model['Attributes']['Selection'] = set(self.selection_attributes())
        intersection = self.entity_attributes().intersection(self.relationship_attributes())
        if len(intersection)!=0:
            raise Exception("ModelError", f"Attribute names should be unique. Found Duplicates: {intersection}")
        self.model['Dependencies'] = set(self.raw_dependencies())
        self.add_implicit_dependencies()
        self.a_graph = nx.DiGraph()
        self.d_graph = nx.DiGraph()
        self.parse_dependencies()
        self.handle_contagion()
        self.functions = None
        
    def set_functions(self, functions):
        self.functions = functions
        
    def handle_contagion(self):
        """
        Automation of contagion dependencies
        TODO: Handle contagion on multiple attributes
        First find topological ordering
        Then, replicate edges from or to Z or Z_L depending on whether parent has contagion
        """
        e_lat_attrs = set()
        r_lat_attrs = set()
        for attr in self.contagion_attributes():
            lat_attr = f'{attr}_L'
            lat_attrs = set(self.latent_attributes())
            lat_attrs.add(lat_attr)
            self.model['Attributes']['Latent'] = lat_attrs
            if attr in self.entity_attributes():
                e_lat_attrs.add(lat_attr)
                lvar=f'E.{lat_attr}'
                c_lvar=f'ERE.{lat_attr}'
                var = f'E.{attr}'
            else:
                r_lat_attrs.add(lat_attr)
                lvar=f'R.{lat_attr}'
                c_lvar=f'RER.{lat_attr}'
                var = f'R.{attr}'
            parents = get_parents(self.d_graph, var)
            for parent in parents:
                parent = Variable(parent)
                self.d_graph.add_edge(str(parent), lvar)
                self.a_graph.add_edge(parent.attribute, lat_attr)
            self.d_graph.add_edge(lvar, var)
            self.a_graph.add_edge(lat_attr, attr)
            self.d_graph.add_edge(c_lvar, var)
                
        ent_attrs = self.entity_attributes().union(e_lat_attrs)
        rel_attrs = self.relationship_attributes().union(r_lat_attrs)
        self.model['Attributes']['Entity'] = ent_attrs
        self.model['Attributes']['Relationship'] = rel_attrs
                            
    def entity_attributes(self):
        return self.model['Attributes']['Entity']
    
    def relationship_attributes(self):
        return self.model['Attributes']['Relationship']
    
    def existence_attribute(self):
        return self.exist
    
    def latent_attributes(self):
        return self.model['Attributes']['Latent']
    
    def selection_attributes(self):
        return self.model['Attributes']['Selection']
    
    def contagion_attributes(self):
        return self.model['Attributes']['Contagion']
    
    def raw_dependencies(self):
        return self.model['Dependencies']
    
    def add_implicit_dependencies(self):
        rel_attrs = self.relationship_attributes()
        if len(rel_attrs) < 2:
            return
        for attr in rel_attrs:
            if attr != self.exist:
                self.model['Dependencies'].add(f'R.{self.exist}->R.{attr}')
    
    def parse_dependencies(self):
        self.dependencies = []
        for dep in self.raw_dependencies():
            dep_obj = Dependency(dep)
            self.dependencies.append(dep_obj)
            self.d_graph.add_edge(str(dep_obj.cause), str(dep_obj.effect))
            self.a_graph.add_edge(str(dep_obj.cause.attribute), str(dep_obj.effect.attribute))
            
