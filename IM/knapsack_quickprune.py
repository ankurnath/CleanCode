from utils import *
from greedy import gain_adjustment,get_gains,calculate_spread

from knapsack_greedy import knapsack_greedy

np.random.seed(0)
random.seed(0)

def qs(graph,gains,node_weights,node_rr_set,RR,budget,delta,eps):

    start = time.time()
    curr_obj = 0
    queries_to_prune = 0
    # pruned_universe=[] 
    a = set()
    # a_start = set() 
    a_start = np.argmax(gains)
    a_s = set()
    covered_rr_set = set ()

    obj_a_s = 0
    uncovered=defaultdict(lambda: True)

    N = graph.number_of_nodes()
    for node in tqdm(graph.nodes()):
        if node_weights[node] >= max_budget:
            continue
        queries_to_prune += 1
        if gains[node]/node_weights[node]>= delta/budget*curr_obj:
            curr_obj+=gains[node]
            # pruned_universe.append(node)
            a.add(node)
            # gain_adjustment(graph,gains,node,uncovered)
            gain_adjustment(gains=gains,node_rr_set=node_rr_set,
                            RR=RR,selected_element=node,
                            covered_rr_set=covered_rr_set)


        ### New addition
        if curr_obj > N/eps*obj_a_s:
            # print('This happened')
            
            # a = a.difference(a_s)
            a.difference_update(a_s)
            a_s = a.copy()

            obj_a_s = calculate_spread(graph=graph,solution=a_s)
            curr_obj = obj_a_s
            queries_to_prune +=1
            
    

    
    a.add(a_start)
    pruned_universe = list(a)
    end= time.time()
    time_to_prune = end-start
    return pruned_universe,queries_to_prune,time_to_prune



def quickprune_multi(dataset,cost_model,max_budget, min_budget,delta ,eps,eta,num_rr):

    # load_graph_file_path=f'../../data/snap_dataset/{dataset}.txt'

    current_folder = os.getcwd()
    load_graph_file_path = os.path.join(current_folder,f'data/snap_dataset/{dataset}.txt')

    graph = load_graph(load_graph_file_path)
    node_weights = generate_node_weights(graph=graph,cost_model=cost_model)
    # node_weights = np.array(list(node_weights.values()))

    gains,node_rr_set,RR = get_gains(graph,num_rr)
    start = time.time()

    # m = int(np.floor (np.log(max_budget/min_budget)/np.log(1+eta)+1))
    # pruned_universe_multi =[]
    # for i in range(m+1):

        # tau = (1+eta)**i * min_budget
    pruned_universe_multi =[]

    high = int(np.log(min_budget/max_budget)/np.log(1-eta) +1 )
    low = int(np.log(max_budget/max_budget)/np.log(1-eta))
    for i in range(low,high+1):
        tau = max_budget*(1-eta)**i
        pruned_universe,queries_to_prune,time_to_prune = qs(graph=graph,
                                                            gains=gains.copy(),
                                                            node_rr_set=node_rr_set,
                                                            RR=RR,
                                                            budget=tau,
                                                            node_weights=node_weights,
                                                            delta=delta,eps=eps)
        pruned_universe_multi +=pruned_universe

    pruned_universe_multi = set (pruned_universe_multi)
    

    end = time.time()

    timetoprune_multi = end-start

    Pg_multi=len(pruned_universe_multi)/graph.number_of_nodes()
    Pg_multi = round(Pg_multi,4)*100
    print("Pg(%):",Pg_multi)
    print('Multi budget Pruned Universe:',len(pruned_universe_multi))
    print("Multi budget Pruned Universe in percentage:",Pg_multi)



    start = time.time()
    pruned_universe_single,_,_ =qs(graph=graph,
                                    gains=gains.copy(),
                                    node_rr_set=node_rr_set,
                                    RR=RR,
                                    budget=max_budget,
                                    node_weights=node_weights,
                                    delta=delta,eps=eps)
    
    

    Pg_single = round(len(pruned_universe_single)/graph.number_of_nodes(),4)*100
    print(f'Single budget Size of Pruned universe:{len(pruned_universe_single)}')
    print("Single budget Pruned Universe in percentage:",Pg_single)

    end= time.time()

    timetoprune_single = end - start
    
    
    



    df = defaultdict(list)
    step = 20
    budgets = list(range(min_budget,max_budget,step)) +[max_budget]
    sprint(budgets)


    for budget in budgets:

        

        start = time.time()

        
        objective_multi_pruned,solution_multi_pruned,queries_multi_pruned = knapsack_greedy     (graph=graph,
                                                                          ground_set = pruned_universe_multi, 
                                                                          num_rr=num_rr,
                                                                          budget = budget, 
                                                                          node_weights = node_weights,
                                                                          gains=gains.copy(),
                                                                          node_rr_set=node_rr_set,
                                                                          RR=RR)
        
        
        
        
        end = time.time()

        time_multi_pruned = end -start

        start = time.time()

        objective_single_pruned,solution_single_pruned,queries_single_pruned =   knapsack_greedy (graph=graph,
                                                                          ground_set = pruned_universe_single, 
                                                                          num_rr=num_rr,
                                                                          budget = budget, 
                                                                          node_weights = node_weights,
                                                                          gains=gains.copy(),
                                                                          node_rr_set=node_rr_set,
                                                                          RR=RR)
        # objective_single_pruned = calculate_spread(graph=graph,solution = solution_single_pruned)

       
        
        end = time.time()
        time_single_pruned = end -start

        start = time.time()
        objective_unpruned,solution_unpruned,queries_unpruned = knapsack_greedy (graph=graph,ground_set =None, 
                                                              num_rr=num_rr,budget = budget, 
                                                              node_weights = node_weights,
                                                              gains=gains.copy(),
                                                              node_rr_set=node_rr_set,
                                                              RR=RR)
        

        
        end = time.time()

        time_unpruned = end- start
        sprint(objective_multi_pruned)
        sprint(objective_single_pruned)
        sprint(objective_unpruned)

        ### TOP-K
        # gains = get_gains(graph,ground_set=None)
        density_gain = {node: gains[node]/node_weights[node] for node in gains}
        # pruned_universe_multi_top_k = [key for key, _ in sorted(density_gain.items(), 
        #                         key=lambda item: item[1], reverse=True) if node_weights[key] <= budget] [:len(pruned_universe_multi)]
        
        pruned_universe_multi_top_k = [key for key, _ in sorted(density_gain.items(), 
                                key=lambda item: item[1], reverse=True) if node_weights[key] <= i] [:len(pruned_universe_multi)]
        
        # pruned_universe_multi_top_k = [key for key, _ in sorted(density_gain.items(), 
        #                         key=lambda item: item[1], reverse=True) ] [:len(pruned_universe_multi)]
        sprint(len(pruned_universe_multi_top_k))
        objective_multi_top_k,solution_multi_top_k,_ = knapsack_greedy(graph=graph,budget=budget, 
                                                    node_weights=node_weights, 
                                                    ground_set=pruned_universe_multi_top_k,
                                                    num_rr=num_rr,
                                                    gains=gains.copy(),
                                                    node_rr_set=node_rr_set,
                                                    RR=RR)
        
        
        # objective_multi_top_k = calculate_spread(graph=graph,solution=solution_multi_top_k)
        # pruned_universe_single_top_k = [key for key, _ in sorted(density_gain.items(), 
        #                         key=lambda item: item[1], reverse=True)if node_weights[key] <= budget ] [:len(pruned_universe_single)]
        # pruned_universe_single_top_k = [key for key, _ in sorted(density_gain.items(), 
        #                         key=lambda item: item[1], reverse=True) [:len(pruned_universe_single)]] 
        
        pruned_universe_single_top_k = [key for key, _ in sorted(density_gain.items(), 
                                key=lambda item: item[1], reverse=True)if node_weights[key] <= i ] [:len(pruned_universe_single)]
        
        sprint(len(pruned_universe_single_top_k))
        
        objective_single_top_k,solution_single_top_k,_ = knapsack_greedy(graph=graph,budget=budget, 
                                                    node_weights=node_weights, 
                                                    ground_set=pruned_universe_single_top_k,
                                                    num_rr=num_rr,
                                                    gains=gains.copy(),
                                                    node_rr_set=node_rr_set,
                                                    RR=RR)
        # objective_single_top_k = calculate_spread(graph=graph,solution=solution_single_top_k)
        
        df['Dataset'].append(dataset)
        df['Budget'].append(budget)
        df['Delta'].append(delta)
        df['Objective Value(Unpruned)'].append(objective_unpruned)
        df['Objective Value Multi(Pruned)'].append(objective_multi_pruned)
        df['Objective Value Single(Pruned)'].append(objective_single_pruned)
        df['Ground Set'].append(graph.number_of_nodes())
        df['Ground set Multi (Pruned)'].append(len(pruned_universe_multi))
        df['Ground set Single (Pruned)'].append(len(pruned_universe_single))
        
        df['Time(Unpruned)'].append(time_unpruned)
        df['Time Multi(Pruned)'].append(time_multi_pruned)
        df['Time Single(Pruned)'].append(time_single_pruned)
        df['Queries (Unpruned)'].append(queries_unpruned)
        df['Queries Multi (pruned)'].append(queries_multi_pruned)
        df['Queries Single (pruned)'].append(queries_single_pruned)

        df['Pruned Ground set Multi(%)'].append(Pg_multi)
        df['Pruned Ground set Single(%)'].append(Pg_single)

        df['Ratio Multi'].append(round(objective_multi_pruned/objective_unpruned,4)*100)
        df['Ratio Single'].append(round(objective_single_pruned/objective_unpruned,4)*100)
        ### TOP-K
        df['Ratio Multi(TOP-K)'].append(round(objective_multi_top_k/objective_unpruned,4)*100)
        df['Ratio Single(TOP-K)'].append(round(objective_single_top_k/objective_unpruned,4)*100)

        df['Queries Multi(%)'].append(round(queries_multi_pruned/queries_unpruned,4)*100)
        df['Queries Single(%)'].append(round(queries_single_pruned/queries_unpruned,4)*100)

        df['TimeRatio(Multi)'].append(time_multi_pruned/time_unpruned)
        df['TimeRatio(Single)'].append(time_single_pruned/time_unpruned)

        df['TimeToPrune(Multi)'].append(timetoprune_multi)
        df['TimeToPrune(Single)'].append(timetoprune_single)




    df = pd.DataFrame(df)
    # print(df)

    # save_folder = f'data/{dataset}/knapsack_multi' 
    save_folder = os.path.join(current_folder,'IM',f'data/knapsack_multi/{dataset}')
    os.makedirs(save_folder,exist_ok=True)
    save_file_path = os.path.join(save_folder,f'quickprune_{cost_model}')
    save_to_pickle(df,save_file_path)


    print(df[['Budget','Ratio Multi','Ratio Multi(TOP-K)','Ratio Single','Ratio Single(TOP-K)']])

    # sprint(set(pruned_universe_multi_top_k)-set(pruned_universe_single_top_k))    
    fontsize = 20
    plt.plot(budgets, df['Ratio Multi'], linestyle='--', marker='o', markersize=20, color='blue', markeredgecolor='black', alpha=0.7, label=f'Multi-Budget {Pg_multi:.2f}%')
    plt.plot(budgets, df['Ratio Single'], linestyle='--', marker='*', markersize=20, color='red', markeredgecolor='black', alpha=0.7, label=f'Single-Budget {Pg_single:.2f}%')
    plt.plot(budgets, df['Ratio Multi(TOP-K)'], linestyle='--', marker='^', markersize=20, color='blue', markeredgecolor='black', alpha=0.7, label=f'Multi-Budget(TOP-K) {Pg_multi:.2f}%')
    plt.plot(budgets, df['Ratio Single(TOP-K)'], linestyle='--', marker='s', markersize=20, color='red', markeredgecolor='black', alpha=0.7, label=f'Single-Budget (TOP-K){Pg_single:.2f}%')
    
    plt.xlabel('Budgets', fontsize=fontsize )
    plt.ylabel('Ratios (%)', fontsize=fontsize)
    plt.title(f' Dataset:{args.dataset} Cost model:{cost_model} Eps:{eps} Eta:{eta} Delta:{delta} \n Max Budget:{max_budget} Min Budget: {min_budget}',fontsize=fontsize)
    plt.legend()

    plt.savefig(os.path.join(save_folder,f'quickprune_{cost_model}.png'), bbox_inches='tight')
    # plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')")
    parser.add_argument("--cost_model",type=str,default='aistats',help='model of node weights')
    parser.add_argument('--max_budget', type = int ,default=100, help = 'Maximum Budget')
    parser.add_argument('--min_budget', type = int ,default=10, help = 'Minimum Budget')

    parser.add_argument("--delta", type=float, default=0.1, help="Delta")
    parser.add_argument("--eps",type =float,default=0.1,help="Epsilon")
    parser.add_argument("--eta",type =float,default=0.5,help="Eta")
    parser.add_argument("--num_rr", type=int, default= 100000  , help="Number of RR sets")

    args = parser.parse_args()

    
    dataset = args.dataset
    cost_model = args.cost_model
    max_budget = args.max_budget
    min_budget = args.min_budget
    delta = args.delta 
    eps = args.eps
    eta = args.eta
    num_rr = args.num_rr

    sprint(dataset)
    sprint(cost_model)
    sprint(max_budget)
    sprint(min_budget)
    sprint(delta)

    quickprune_multi(dataset=dataset, cost_model=cost_model , 
                      max_budget=max_budget, 
                      min_budget =min_budget,
                      delta= delta ,
                      eps=eps,
                      num_rr=num_rr,
                      eta=eta)

