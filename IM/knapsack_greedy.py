from utils import *
from greedy import get_gains, gain_adjustment,calculate_spread



def knapsack_greedy(graph,ground_set , budget ,node_weights,
                    gains=None,node_rr_set=None,RR=None,num_rr=None):


    if gains is None or node_rr_set is None or RR is None:
        gains,node_rr_set,RR = get_gains(graph=graph,num_rr=num_rr)

    if ground_set:
        gains= {node:gains[node] for node in ground_set if node in gains}

    # sprint(gains)

    # gains = {}

    # get max singleton

    max_singleton = None

    max_singleton_gain = 0

    number_of_queries = len(gains)

    # sprint(len(gains))

    # only taking element in ground set
    for element in gains:
        # try:
        if node_weights[element]<= budget and gains[element] > max_singleton_gain:
            max_singleton = element
            max_singleton_gain = gains [element]
        # except:
        #     sprint(element)
        #     sprint(node_weights[element])
        #     sprint(gains[element])
        #     raise ValueError


    constraint = 0
    N = len (gains)

    covered_rr_set = set ()
    solution = []

    objective_value = 0
    
    while gains:
        number_of_queries+= len(gains)
        max_gain_ratio = 0

        selected_element = None

        for element in gains:
            if gains[element]/node_weights[element]> max_gain_ratio:
                max_gain_ratio = gains[element]/node_weights[element]
                selected_element = element

        if max_gain_ratio == 0 :

            break

        if node_weights[selected_element]+constraint <= budget:

            solution.append(selected_element)
            objective_value += gains[selected_element]
            gain_adjustment(gains=gains,node_rr_set=node_rr_set,RR=RR,
                            selected_element=selected_element,covered_rr_set=covered_rr_set)
            constraint += node_weights[selected_element]


        if constraint == budget:
            break

        gains.pop(selected_element)

    # print('Number of queries:',number_of_queries)
        
    if calculate_spread(graph,solution)< calculate_spread(graph,solution=[max_singleton]) :
        solution = [ max_singleton ]

    return objective_value,solution,number_of_queries

