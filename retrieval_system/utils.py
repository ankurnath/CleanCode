
from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from numba import njit,prange
from argparse import ArgumentParser
from collections import defaultdict

import pandas as pd
import pickle
import random
from tqdm import tqdm
import heapq
import os

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












    





    pass



@njit(fastmath=True,parallel=True)
def QS(similarity, costs, delta, budget):
    """
    Maximizes the ground set based on similarity and cost constraints.

    Args:
        similarity (np.ndarray): NxN matrix representing pairwise similarity scores between text elements.
        costs (np.ndarray): 1D array representing the cost associated with each text element.
        delta (float): A constant to regulate the minimum gain to cost ratio.
        budget (float): The available budget for selecting elements.

    Returns:
        np.ndarray: A binary array indicating which elements are selected in the ground set (1 if selected, 0 if not).
    """
    
    # Number of elements in the set
    N = len(similarity)
    print('Size of unpruned ground set',N)
    
    # Current objective value
    curr_obj = 0
    
    # Maximum similarity values for each element
    max_similarity = np.zeros(N)
    
    # Ground set to keep track of selected elements (0: not selected, 1: selected)
    ground_set = np.zeros(N)
    
    # Loop through all elements to consider them for the ground set
    for element in range(N):
        obj_val = 0
        
        # Calculate the objective value by updating the maximum similarity
        for i in prange(N):
            obj_val += max(max_similarity[i], similarity[i, element])

        # Gain is the increase in the objective value
        gain = obj_val - curr_obj
        
        # Check if the gain-to-cost ratio meets the threshold based on delta and budget
        if gain / costs[element] >= delta / budget * curr_obj:
            # Update the current objective value with the gain
            curr_obj += gain
            
            # Mark the element as selected in the ground set
            ground_set[element] = 1
            for i in range(N):
                max_similarity[i] = max(max_similarity[i], similarity[i, element])

    print('Size of pruned ground set',ground_set.sum())
    return ground_set


@njit(fastmath=True,parallel=True)
def facility_location(similarity,costs,budget,ground_set):
    # N= 25000
    N = len(similarity)

    max_obj = 0
    total_cost = 0
    solution_sparse = np.zeros(N)

    max_similarity = np.zeros(N)

    while total_cost < budget:

        max_element = -1
        obj_val = np.zeros(N)

        for element in prange(N):
            if solution_sparse[element] == 0 and ground_set[element] ==1 and costs[element]+total_cost <=budget:


                for i in range(N):
                    obj_val[element] += max(max_similarity[i],similarity[i,element])
         

        max_element = np.argmax(obj_val)

        if obj_val[max_element] == max_obj:
            break

        else:
            solution_sparse[max_element] = 1
            total_cost += costs[max_element]
            for i in range(N):
                max_similarity[i] = max(max_similarity[i],similarity[i,max_element])
            
            max_obj = obj_val[max_element]

    # print(max_obj)
    # print(solution_sparse.sum())
    return max_obj,solution_sparse