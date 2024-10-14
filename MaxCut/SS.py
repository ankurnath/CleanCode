from utils import *
from greedy import *
import heapq


def SS(dataset,r,c,budget):

    current_folder = os.getcwd()
    file_path= os.path.join(current_folder,f'data/snap_dataset/{args.dataset}.txt')
    graph = load_graph(file_path=file_path)


    queries_to_prune = 0

    start = time.time()
    pruned_universe=set()
    universe=list(graph.nodes())
    n=graph.number_of_nodes()
    
    while len(universe)> r*np.log2(n):
        print('Size of universe:',len(universe))
        U=random.sample(universe,int(r*np.log2(n)))
        universe = set(universe)
        for node in tqdm(U):
            universe.remove(node)
        # universe = list(universe)
        U=set(U)
        pruned_universe=pruned_universe.union(U)


        universe_gain=calculate_obj(graph,universe) # f(V)
        queries_to_prune += 1

        # for v in universe:

        universe_u_gain = {} # f(V U u)
        u_gain = {} # f(u)
        # get all neighbors 
        
        
        for u in tqdm(U):
            universe.add(u)
            universe_u_gain[u] = calculate_obj (graph ,universe)
            queries_to_prune += 1
            universe.remove(u)
            u_gain[u] = calculate_obj (graph , [u])
            queries_to_prune += 1


        lst = []

        for v in tqdm(universe):

            w=float('inf')
            
            # for u in graph.neighbors(v):
                
            for u in U:
                # universe_copy=universe.copy()
                # universe_copy.append(u)
                
                local_gain = calculate_obj(graph,[u,v])-u_gain[u] # f(v U u) -f(u)
                queries_to_prune += 1
                # print(local_gain)

                global_gain = universe_u_gain[u]-universe_gain
                w=min(w,local_gain-global_gain)

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

    print('time elapsed to pruned',time_to_prune)

    ##################################################################

    Pg=len(pruned_universe)/graph.number_of_nodes()
    start = time.time()
    objective_unpruned,queries_unpruned,solution_unpruned= greedy(graph,budget)
    end = time.time()
    time_unpruned = round(end-start,4)
    print('Elapsed time (unpruned):',round(time_unpruned,4))

    start = time.time()
    objective_pruned,queries_pruned,solution_pruned = greedy(graph=graph,budget=budget,ground_set=pruned_universe)
    end = time.time()
    time_pruned = round(end-start,4)
    print('Elapsed time (pruned):',time_pruned)
    
    
    objective_unpruned = calculate_obj(graph,solution_unpruned)
    objective_pruned = calculate_obj(graph,solution_pruned)
    
    ratio = objective_pruned/objective_unpruned


    print('Performance of SS')
    print('Size Constraint,k:',budget)
    print('Size of Ground Set,|U|:',graph.number_of_nodes())
    print('Size of Pruned Ground Set, |Upruned|:', len(pruned_universe))
    print('Pg(%):', round(Pg,4)*100)
    print('Ratio:',round(ratio,4)*100)
    print('Queries:',round(queries_pruned/queries_unpruned,4)*100)


    
    

    save_folder = os.path.join(current_folder,'MaxCut',f'data/{dataset}')
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
              'TimeToPrune':time_to_prune,
              

              }

   
    df = pd.DataFrame(df,index=[0])
    save_to_pickle(df,save_file_path)
    print(df)
    

    ###################################################################################################


   

    




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument( "--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument( "--r", type=float, default=8, help="r" )
    parser.add_argument( "--c", type=float, default=8, help="c" )
    parser.add_argument("--budget", type=int,default=100, help="Budgets")

    args = parser.parse_args()

    SS(dataset=args.dataset,r=args.r,c=args.c,budget=args.budget)

    