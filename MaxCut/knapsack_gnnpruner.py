import os
import torch

import numpy as np
from utils import *
from argparse import ArgumentParser
from torch_geometric.utils.convert import  from_networkx
from greedy import greedy
from collections import defaultdict
# from knapsack_numba_greedy import knapsack_numba_greedy
from DLA_numba import DLA

import pandas as pd

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument( "--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument( "--budget", type= float , default= 100, help="Budget" )
    parser.add_argument("--cost_model",type= str, default= 'aistats', help = 'model of node weights')
    args = parser.parse_args()


    dataset = args.dataset
    budget = args.budget
    current_folder = os.getcwd()
    load_graph_file_path = os.path.join(current_folder,f'data/train/{dataset}')
    

    train_graph=load_graph(load_graph_file_path)
    
    data= from_networkx(train_graph)


    
    cost_model = args.cost_model
    node_weights = generate_node_weights(graph=train_graph,cost_model=cost_model)
    _,solution= DLA(graph=train_graph,budget=args.budget,node_weights=node_weights)


    mapping = dict(zip(train_graph.nodes(), range(train_graph.number_of_nodes())))
    train_mask = torch.tensor([mapping[node] for node in solution], dtype=torch.long)
    y=torch.zeros(train_graph.number_of_nodes(),dtype=torch.long)


    for node in solution:
        y[mapping[node]]=1



    data.y=y
    num_features=3

    # x=[graph.degree(node) for node in graph.nodes()] 
    x = [[node_weights[node] for node in train_graph.nodes()],[train_graph.degree(node) for node in train_graph.nodes()],
         [train_graph.degree(node)/node_weights[node] for node in train_graph.nodes()]]
    # data.x = torch.rand(size=(graph.number_of_nodes(),1))
    data.x = torch.FloatTensor(x).permute(1,0)

    # torch.FloatTensor

    # print(data.x.shape)


    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv

    class GCN(torch.nn.Module):
        def __init__(self, hidden_channels):
            super(GCN, self).__init__()
            torch.manual_seed(12345)
            self.conv1 = GCNConv(num_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, 2)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = self.conv2(x, edge_index)
            return x

    model = GCN(hidden_channels=16)

    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()

    best_loss = float('inf')  # Initialize the best training loss to infinity

    for epoch in tqdm(range(1,1000)):

        out = model(data.x, data.edge_index)  # Perform a single forward pass.

        # mask=torch.cat([train_mask,torch.randint(graph.number_of_nodes())],axis=0)
        mask = torch.cat([train_mask, torch.randint(0, train_mask.size(0), (train_mask.size(0),))], dim=0)
        # print('Mask size',train_mask.shape)
        # print('Mask size',mask.shape)

        # print(torch.sum(data.y[mask]))
        loss = criterion(out[mask], data.y[mask])  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

        # Save the best model if training loss improves
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), 'best_model.pth')  # Save the model's state dictionary
            # print(f"Epoch {epoch}: Training loss improved to {best_loss:.4f}. Model saved.")
        
    model.load_state_dict(torch.load('best_model.pth'))

    model.eval()


    
    current_folder = os.getcwd()
    load_graph_file_path = os.path.join(current_folder,f'data/snap_dataset/{dataset}.txt')

    test_graph = load_graph(load_graph_file_path)
    test_data = from_networkx(test_graph)
    test_node_weights = generate_node_weights(graph=test_graph,cost_model=cost_model)



    x = [[test_node_weights[node] for node in test_graph.nodes()],[test_graph.degree(node) for node in test_graph.nodes()],
         [test_graph.degree(node)/test_node_weights[node] for node in test_graph.nodes()]]
    # test_data.x =torch.rand(size=(test_graph.number_of_nodes(),1)) 
    test_data.x = torch.FloatTensor(x).permute(1,0)

    start = time.time()
    out = model(test_data.x, test_data.edge_index)
    pred = out.argmax(dim=1).cpu().numpy()  # Use the class with highest probability.
    indices = np.where(pred == 1)[0]
    reverse_mapping = dict(zip(range(test_graph.number_of_nodes()),test_graph.nodes()))
    pruned_universe = [reverse_mapping[node] for node in indices]
    # print(pruned_universe)
    end= time.time()

    time_to_prune = end-start

    print('time elapsed to pruned',time_to_prune)

    graph = test_graph

    ##################################################################

    Pg=len(pruned_universe)/test_graph.number_of_nodes()
    start = time.time()
    objective_unpruned,solution_unpruned= DLA(graph=test_graph,budget=args.budget,
                                                                                 node_weights=test_node_weights)
    end = time.time()
    time_unpruned = round(end-start,4)
    print('Elapsed time (unpruned):',round(time_unpruned,4))

    start = time.time()
    objective_pruned,solution_pruned = DLA(graph=test_graph,budget=args.budget,
                                                                            node_weights=test_node_weights,
                                                                            ground_set=pruned_universe)
    end = time.time()
    time_pruned = round(end-start,4)
    print('Elapsed time (pruned):',time_pruned)
    
    
    ratio = objective_pruned/objective_unpruned


    print('Performance of GNNpruner')
    print('Size Constraint,k:',budget)
    print('Size of Ground Set,|U|:',test_graph.number_of_nodes())
    print('Size of Pruned Ground Set, |Upruned|:', len(pruned_universe))
    print('Pg(%):', round(Pg,4)*100)
    print('Ratio:',round(ratio,4)*100)
    # print('Queries:',round(queries_pruned/queries_unpruned,4)*100)


    save_folder =  os.path.join(current_folder,'MaxCut',f'data/{dataset}/Knapsack_GNN') 
    os.makedirs(save_folder,exist_ok=True)
    save_file_path = os.path.join(save_folder,f'GNNpruner_{cost_model}')

    df ={      'Dataset':dataset,'Budget':budget,'Objective Value(Unpruned)':objective_unpruned,
              'Objective Value(Pruned)':objective_pruned ,'Ground Set': test_graph.number_of_nodes(),
              'Ground set(Pruned)':len(pruned_universe), 
            #   'Queries(Unpruned)': queries_unpruned,
              'Time(Unpruned)':time_unpruned,
              'Time(Pruned)': time_pruned,
            #   'Queries(Pruned)': queries_pruned,
                'Pruned Ground set(%)': round(Pg,4)*100,
              'Ratio(%)':round(ratio,4)*100, 
            #   'Queries(%)': round(queries_pruned/queries_unpruned,4)*100,
              'TimeRatio': time_pruned/time_unpruned,
              'TimeToPrune':time_to_prune

              }

   
    df = pd.DataFrame(df,index=[0])
    save_to_pickle(df,save_file_path)
    print(df)






