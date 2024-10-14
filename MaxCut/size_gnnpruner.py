import torch
from utils import *
from torch_geometric.utils.convert import  from_networkx
from greedy import greedy


import pandas as pd

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument( "--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument( "--budget", type= int , default= 100, help="Budget" )
    args = parser.parse_args()


    dataset = args.dataset
    budget = args.budget
    current_folder = os.getcwd()
    load_graph_file_path = os.path.join(current_folder,f'data/train/{dataset}')



    train_graph=load_from_pickle(load_graph_file_path)
    data= from_networkx(train_graph)


    _,_,solution=greedy(graph=train_graph,budget=budget,ground_set=None)


    mapping = dict(zip(train_graph.nodes(), range(train_graph.number_of_nodes())))
    train_mask = torch.tensor([mapping[node] for node in solution], dtype=torch.long)
    y=torch.zeros(train_graph.number_of_nodes(),dtype=torch.long)


    for node in solution:
        y[mapping[node]]=1



    data.y=y
    num_features=1

    x=[train_graph.degree(node) for node in train_graph.nodes()]
    data.x = torch.rand(size=(train_graph.number_of_nodes(),1))


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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device=device)
    data.to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()

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
        


    model.eval()

    current_folder = os.getcwd()
    load_graph_file_path = os.path.join(current_folder,f'data/snap_dataset/{dataset}.txt')

    test_graph = load_graph(load_graph_file_path)
    test_data = from_networkx(test_graph)

    test_data.x =torch.rand(size=(test_graph.number_of_nodes(),1))

    start = time.time()
    test_data = test_data.to(device)
    out = model(test_data.x, test_data.edge_index)
    pred = out.argmax(dim=1).cpu().numpy()  # Use the class with highest probability.
    indices = np.where(pred == 1)[0]
    reverse_mapping = dict(zip(range(test_graph.number_of_nodes()),test_graph.nodes()))
    pruned_universe = [reverse_mapping[node] for node in indices]
    end= time.time()

    time_to_prune = end-start

    print('time elapsed to pruned',time_to_prune)

    

    ##################################################################

    Pg=len(pruned_universe)/test_graph.number_of_nodes()
    start = time.time()
    objective_unpruned,queries_unpruned,solution_unpruned= greedy(test_graph,budget)
    end = time.time()
    time_unpruned = round(end-start,4)
    print('Elapsed time (unpruned):',round(time_unpruned,4))

    start = time.time()
    objective_pruned,queries_pruned,solution_pruned = greedy(graph=test_graph,budget=budget,ground_set=pruned_universe)
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
    print('Queries:',round(queries_pruned/queries_unpruned,4)*100)


    save_folder = f'MaxCut/data/{dataset}'
    os.makedirs(save_folder,exist_ok=True)
    save_file_path = os.path.join(save_folder,'GNNpruner')

    

    df ={     'Dataset':dataset,
              'Budget':budget,
              'Objective Value(Unpruned)':objective_unpruned,
              'Objective Value(Pruned)':objective_pruned ,
              'Ground Set': test_graph.number_of_nodes(),
              'Ground set(Pruned)':len(pruned_universe), 
              'Queries(Unpruned)': queries_unpruned,
              'Time(Unpruned)':time_unpruned,
              'Time(Pruned)': time_pruned,
              'Queries(Pruned)': queries_pruned, 
              'Pruned Ground set(%)': round(Pg,4)*100,
              'Ratio(%)':round(ratio,4)*100, 
              'Queries(%)': round(queries_pruned/queries_unpruned,4)*100,
              'TimeRatio': time_pruned/time_unpruned,
              'TimeToPrune':time_to_prune,
              


              }

   
    df = pd.DataFrame(df,index=[0])
    save_to_pickle(df,save_file_path)
   





