
from utils import *
from imm import *
from greedy import *
import heapq








def SS(dataset,r,c,num_rr,budget):

    
    # file_path=f'../../data/snap_dataset/{args.dataset}.txt' 

    current_folder = os.getcwd()
    load_graph_file_path = os.path.join(current_folder,f'data/snap_dataset/{dataset}.txt')

    graph = load_graph(file_path=load_graph_file_path)

    queries_to_prune = 0


    start = time.time()
    pruned_universe=set()
    universe=list(graph.nodes())
    n=graph.number_of_nodes()

    gains,node_rr_set,RR = get_gains(graph,num_rr)

    queries_to_prune += len(gains)
    def calculate_spread(graph,solution):

        covered = set()

        for node in solution:
            for index in node_rr_set[node]:
                covered.add(index)

        return len(covered)/len(RR)*graph.number_of_nodes()

    while len(universe)> r*np.log2(n):
        print('Size of universe:',len(universe))
        U=random.sample(universe,int(r*np.log2(n)))
        universe = set(universe)
        for node in tqdm(U):
            universe.remove(node)
        # universe = list(universe)
        U=set(U)
        pruned_universe=pruned_universe.union(U)


        # universe_gain= calculate_obj(graph,universe) # f(V) 
        universe_gain = calculate_spread(graph,universe)
        queries_to_prune += 1

        # for v in universe:

        universe_u_gain = {} # f(V U u)
        u_gain = {} # f(u)
        # get all neighbors 
        
        
        for u in tqdm(U):
            universe.add(u)
            # universe_u_gain[u] = calculate_obj (graph ,universe)
            universe_u_gain[u] = calculate_spread (graph ,universe)
            queries_to_prune += 1
            universe.remove(u)
            # u_gain[u] = calculate_obj (graph , [u])
            u_gain[u] = calculate_spread (graph , [u])
            queries_to_prune += 1


        lst = []

        for v in tqdm(universe):

            w=float('inf')
            
            # for u in graph.neighbors(v):
                
            for u in U:
                # universe_copy=universe.copy()
                # universe_copy.append(u)
                
                # local_gain = calculate_obj(graph,[u,v])-u_gain[u] # f(v U u) -f(u)
                local_gain = calculate_spread(graph,[u,v])-u_gain[u] # f(v U u) -f(u)
                queries_to_prune += 1
                # print(local_gain)

                global_gain = universe_u_gain[u]-universe_gain
                w = min(w,local_gain-global_gain)

            lst.append((w,v))

        remove_nodes=heapq.nsmallest(int((1-1/np.sqrt(c))*len(universe)), lst)
        # print(remove_nodes)
        universe = set(universe)
        for w,node in tqdm(remove_nodes):
            # if w>0:
            #     print(w)
            universe.remove(node)
            # universe.re
        universe = list(universe)

        

    pruned_universe=pruned_universe.union(set(universe))

    end= time.time()

    time_to_prune = end-start

    print('Number of queries needed to be pruned',queries_to_prune)

    print('time elapsed to pruned',time_to_prune)

    subgraph = make_subgraph(graph,pruned_universe)

    ##################################################################

    Pg=len(pruned_universe)/graph.number_of_nodes()
    start = time.time()
    # solution_unpruned, _ = imm(graph=graph,seed_size=budget,seed=0)
    solution_unpruned = imm(graph=graph,seed_size=budget,seed=0)
    queries_unpruned  = budget/2 * (2*graph.number_of_nodes() - budget +1) 
    end = time.time()


    # sprint([graph.degree(node) for node in solution_unpruned])
    
    time_unpruned = round(end-start,4)
    print('Elapsed time (unpruned):',round(time_unpruned,4))

    start = time.time()
    # solution_pruned, _ = imm(graph=subgraph,seed_size=budget, seed=0)
    solution_pruned = imm(graph=subgraph,seed_size=budget, seed=0)
    queries_pruned  = budget/2 * (2*len(pruned_universe) - budget +1) 
    # sprint([graph.degree(node) for node in solution_pruned])
    end = time.time()
    time_pruned = round(end-start,4)
    print('Elapsed time (pruned):',time_pruned)

    objective_pruned = calculate_spread(graph=graph,solution=solution_pruned)
    objective_unpruned = calculate_spread(graph=graph,solution=solution_unpruned)

    sprint(objective_pruned)
    sprint(objective_unpruned)
    ratio = objective_pruned/objective_unpruned


    print('Performance of SS')
    print('Size Constraint,k:',budget)
    print('Size of Ground Set,|U|:',graph.number_of_nodes())
    print('Size of Pruned Ground Set, |Upruned|:', len(pruned_universe))
    print('Pg(%):', round(Pg,4)*100)
    print('Ratio:',round(ratio,4)*100)


    save_folder = os.path.join(current_folder,'IM',f'data/{dataset}')
    os.makedirs(save_folder,exist_ok=True)
    save_file_path = os.path.join(save_folder,'SS')

    df ={     'Dataset':dataset,
              'Budget':budget,
              'r':r,
              'c':c,
              'QueriesToPrune': queries_to_prune,
              'Objective Value(Unpruned)':objective_unpruned,
              'Objective Value(Pruned)':objective_pruned ,
              'Ground Set': graph.number_of_nodes(),
              'Ground set(Pruned)':len(pruned_universe), 
              'Queries(Unpruned)': queries_unpruned,
              'Time(Unpruned)':time_unpruned,
              'Time(Pruned)': time_pruned,
              'Queries(Pruned)': queries_pruned, 
              'Pruned Ground set(%)': round(Pg,4)*100,
              'Ratio(%)':round(ratio,4)*100, 
              'Queries(%)': round(queries_pruned/queries_unpruned,4)*100,
              'TimeRatio': time_pruned/time_unpruned,
              'TimeToPrune':time_to_prune

              }

   
    df = pd.DataFrame(df,index=[0])
    save_to_pickle(df,save_file_path)
    print(df)

    ##################################################################
   

    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument( "--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument( "--r", type=float, default=8, help="r" )
    parser.add_argument( "--c", type=float, default=8, help="c" )
    parser.add_argument("--budget", type=int,default=100, help="Budgets")
    parser.add_argument("--num_rr", type=int, default= 100000  , help="Number of RR sets")

    args = parser.parse_args()

    dataset = args.dataset
    r = args.r 
    c = args.c 
    budget = args.budget
    num_rr = args.num_rr

    sprint(dataset)
    sprint(r)
    sprint(c)
    sprint(budget)
    sprint(num_rr)

    
    SS (dataset=dataset,r=r,c=c,budget=budget,num_rr=num_rr)
