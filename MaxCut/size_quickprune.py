from utils import *
from greedy import greedy,gain_adjustment,get_gains


def qs(graph,budget,delta,eps):
    start = time.time()
    gains = get_gains(graph,ground_set=None)
    curr_obj = 0
    queries_to_prune = 0
    a = set()
    a_s = set()
    a_start = max(gains, key=gains.get)
    
    
    obj_a_s = 0
    uncovered=defaultdict(lambda: True)

    N = graph.number_of_nodes()
    for node in tqdm(graph.nodes()):
        queries_to_prune += 1
        if gains[node]>= delta/budget*curr_obj:
            curr_obj+=gains[node]
            a.add(node)
            gain_adjustment(graph,gains,node,uncovered)


        ### New addition
        if curr_obj > N/eps*obj_a_s:
            a.difference_update(a_s)
            a_s = a.copy()

            obj_a_s = calculate_obj(graph=graph,solution=a_s)
            curr_obj = obj_a_s
            queries_to_prune +=1
            
    end= time.time()

    time_to_prune = end-start

    print('time elapsed to pruned',time_to_prune)
    a.add(a_start)
    pruned_universe = list(a)
    return pruned_universe,queries_to_prune,time_to_prune


def quickprune(dataset,max_budget,min_budget,delta,eps,eta):

  
    
    current_folder = os.getcwd()
    load_graph_file_path = os.path.join(current_folder,f'data/snap_dataset/{dataset}.txt')
    
    graph = load_graph(load_graph_file_path)

    pruned_universe = []
    queries_to_prune = 0
    time_to_prune = 0

    high = int(np.log(min_budget/max_budget)/np.log(1-eta) +1 )
    low = int(np.log(max_budget/max_budget)/np.log(1-eta))

    for i in range(low,high+1):
        tau = max_budget*(1-eta)**i

        temp_pruned_universe,temp_queries_to_prune,temp_time_to_prune = qs(graph=graph,
                                                                           budget=tau,
                                                                           delta=delta,
                                                                           eps=eps)
        
        pruned_universe += temp_pruned_universe
        queries_to_prune +=temp_queries_to_prune
        time_to_prune +=temp_time_to_prune 

    # sprint(len(pruned_universe))
    pruned_universe = set(pruned_universe)
    

    ##################################################################

    Pg=len(pruned_universe)/graph.number_of_nodes()
    start = time.time()
    objective_unpruned,queries_unpruned,solution_unpruned= greedy(graph,max_budget)
    # assert len(solution_unpruned) <= budget, 'Solution from  ground set exceeds the budget'
    # assert objective_unpruned == calculate_obj(graph=graph,solution=solution_unpruned)
    
    end = time.time()
    time_unpruned = round(end-start,4)
    print('Elapsed time (unpruned):',round(time_unpruned,4))

    start = time.time()
    objective_pruned,queries_pruned,solution_pruned = greedy(graph=graph,budget=max_budget,ground_set=pruned_universe)
    # assert len(solution_pruned) <= budget, 'Solution from pruned ground set exceeds the budget'
    # assert objective_pruned == calculate_obj(graph=graph,solution=solution_pruned)
    end = time.time()
    time_pruned = round(end-start,4)
    print('Elapsed time (pruned):',time_pruned)
    ratio = objective_pruned/objective_unpruned


    print('Performance of quickprune')
    print('Size Constraint,k:',max_budget)
    print('Size of Ground Set,|U|:',graph.number_of_nodes())
    print('Size of Pruned Ground Set, |Upruned|:', len(pruned_universe))
    print('Pg(%):', round(Pg,4)*100)
    print('Ratio:',round(ratio,4)*100)
    print('Queries:',round(queries_pruned/queries_unpruned,4)*100)


    # save_folder = f'data/{dataset}'
    save_folder = os.path.join(current_folder,'MaxCut',f'data/{dataset}')
    os.makedirs(save_folder,exist_ok=True)
    save_file_path = os.path.join(save_folder,'quickprune')


    

        



    df ={     'Dataset':dataset,
              'Max Budget': max_budget,
              'Min Budget': min_budget,
              'Delta':delta,
              'eps':eps,
              'eta':eta,
              'QueriesToPrune': queries_to_prune,
              'Objective Value(Unpruned)':objective_unpruned,
              'Objective Value(Pruned)':objective_pruned ,
              'Ground Set': graph.number_of_nodes(),
              'Ground set(Pruned)':len(pruned_universe), 
              'Queries(Unpruned)': queries_unpruned,'Time(Unpruned)':time_unpruned,
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
    

    ###################################################################################################

   
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')")
    parser.add_argument('--max_budget', type = int ,default=100, help = 'Maximum Budget')
    parser.add_argument('--min_budget', type = int ,default=10, help = 'Minimum Budget')
    
    
    parser.add_argument("--delta", type=float, default=0.1, help="Delta")
    parser.add_argument("--eps", type=float, default=0.1, help="eps")
    parser.add_argument("--eta",type =float,default=0.5,help="Eta")


    
    

    args = parser.parse_args()

    dataset = args.dataset
    max_budget = args.max_budget
    min_budget = args.min_budget
    delta = args.delta 
    eps = args.eps
    eta = args.eta

    sprint(dataset)
    sprint(max_budget)
    sprint(min_budget)
    sprint(delta)
    sprint(eps)
    sprint(eta)

    
    quickprune(dataset=dataset,
                max_budget=max_budget,
                min_budget=min_budget,
                delta=delta,
                eps=eps,
                eta=eta)
