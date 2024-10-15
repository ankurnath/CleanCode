from utils import *

from collections import deque
from imm import *


def calculate_spread(graph,solution,mc=10000):
    sprint('Calculating spread')
    weight = 0.01    
    print(f'Default weight = {weight}')

    spread = 0

    for _ in tqdm(range(mc)):

        activated_nodes = set(solution)
        queue = deque(solution)

        while queue:
            node = queue.popleft()

            for neighbor in graph.neighbors(node):
                if neighbor not in activated_nodes and random.random() < weight:
                    activated_nodes.add(neighbor)
                    queue.append(neighbor)
        
        spread += len(activated_nodes)

    return spread/mc





    

def get_gains(graph,num_rr):

    graph_ = get_graph(graph)


    # tracks what nodes cover each reversible sets
    RR = []

    worker = []
    worker_num =NUM_PROCESSORS
    create_worker(num =worker_num, worker = worker,  model = 'IC', graph_=graph_)

    for ii in range(worker_num):
            worker[ii].inQ.put(num_rr / worker_num)
    for w in worker:
        R_list = w.outQ.get()
        RR += R_list

    finish_worker(worker)

    

    gains = defaultdict(int) 
    # Keep tracks for what node covers which RR sets
    node_rr_set = defaultdict(list)

    for j,rr in enumerate(RR):
        for rr_node in rr:
            gains[rr_node]+=1
            node_rr_set[rr_node].append(j)

    
    return gains,node_rr_set,RR



def gain_adjustment(gains,node_rr_set,RR,selected_element,covered_rr_set):
     
    for index in node_rr_set[selected_element]:
        if index not in covered_rr_set:
            covered_rr_set.add(index)
            for rr_node in RR[index]:
                if rr_node in gains:
                    gains[rr_node]-=1

    
    assert gains[selected_element] == 0, 'gains adjustment error'















    
