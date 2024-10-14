import pickle
import networkx as nx
from argparse import ArgumentParser
import networkx as nx
import numpy as np
import random
import os
import logging

def load_graph(file_path):

    try:
        graph = nx.read_edgelist(file_path, create_using=nx.Graph(), nodetype=int)

    except:
        f = open(file_path, mode="r")
        lines = f.readlines()
        edges = []

        for line in lines:
            line = line.split()
            if line[0].isdigit():
                edges.append([int(line[0]), int(line[1])])
        graph = nx.Graph()
        graph.add_edges_from(edges)
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    return graph
def load_from_pickle(file_path):
    """
    Load data from a pickle file.

    Parameters:
    - file_path: The path to the pickle file.

    Returns:
    - loaded_data: The loaded data.
    """
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    print(f'Data has been loaded from {file_path}')
    return loaded_data


def save_to_pickle(data, file_path):
    """
    Save data to a pickle file.

    Parameters:
    - data: The data to be saved.
    - file_path: The path to the pickle file.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
    print(f'Data has been saved to {file_path}')