from utils import *



def flatten_graph(graph:nx.Graph):
    flat_adj_matrix = []
    
    n = graph.number_of_nodes()
    start = [0 for _ in range(n)]
    end = [0 for _ in range(n)]
    adj_list_dict = nx.to_dict_of_lists(graph)

    

    for node, neighbors in adj_list_dict.items():
        start[node] = len(flat_adj_matrix)
        end[node] = start[node] + len(neighbors)
        flat_adj_matrix += neighbors
    return np.array(flat_adj_matrix), np.array(start), np.array(end)

@njit
def get_gains(start,end):

    N = len(start)
    gains = np.zeros(N)

    for node in range(N):
        gains[node]=end[node]-start[node]

    return gains


@njit
def calculate_obj(adj_list,start,end,solution):

    N = len(start)

    objVal =0

    for node in range(N):
        if solution[node] ==1:
            for neighbor in adj_list[start[node]:end[node]]:
                if solution[node] != solution[neighbor]:
                    objVal +=1

    return objVal

    
# gains[neighbor]+=(2*spins[neighbor]-1)*(2-4*spins[selected_element])


@njit
def gain_adjustment(adj_list,start,end,gains,selected_element,spins):

    gains[selected_element] = - gains[selected_element]

    for neighbor in adj_list[start[selected_element]:end[selected_element]]:

        gains[neighbor] += (2*spins[neighbor]-1)*(2-4*spins[selected_element])

    spins[selected_element] = 1 - spins[selected_element]


@njit
def LA(adj_list,start,end,gains,node_weights,budget,mask):


    max_singleton = -1
    max_gain  = 0

    N = len(start)

    for node in range(N):
        if mask[node]==1 and node_weights[node] <= budget and max_gain <= gains[node]:
            max_gain = gains[node]
            max_singleton = node

    # print(max_gain)

    gains_X = gains.copy()
    gains_Y = gains.copy()

    # Initialize spins and objectives
    spins_X = np.zeros(N)
    spins_Y = np.zeros(N)

    obj_X = 0
    obj_Y = 0

    # Initialize selected nodes lists 
    X = np.zeros(shape=(N,),dtype=np.int32)
    Y = np.zeros(shape=(N,),dtype=np.int32)

    size_X = 0
    size_Y = 0

    for node in range(N):
    # for node in range(3):
        if mask[node]==1 and node_weights[node]<=budget/2:
            # sprint(node)
            # sprint(gains_X[node])
            # sprint(gains_Y[node]) 
            # Calculate density gains once
            density_gain_X = gains_X[node] / node_weights[node]
            density_gain_Y = gains_Y[node] / node_weights[node]
            # sprint(density_gain_X)
            # sprint(density_gain_Y)

            # Calculate current objectives per budget ratio
            obj_ratio_X = obj_X / budget
            obj_ratio_Y = obj_Y / budget

            # sprint(obj_ratio_X)
            # sprint(obj_ratio_Y)

            # Decide where to add the node
            if density_gain_X >= obj_ratio_X and density_gain_Y >= obj_ratio_Y:
                if density_gain_X >= density_gain_Y:
                    obj_X += gains_X[node]
                    gain_adjustment(adj_list=adj_list,start=start,end=end,gains=gains_X,
                                    selected_element=node, spins=spins_X)
                    
                                       
                    X[size_X] = node
                    size_X +=1 
                else:
                    obj_Y += gains_Y[node]
                    gain_adjustment(adj_list=adj_list,start=start,end=end, gains=gains_Y, 
                                    selected_element=node, spins=spins_Y)
                    Y[size_Y] = node
                    size_Y +=1

            elif density_gain_X >= obj_ratio_X:
                obj_X += gains_X[node]
                gain_adjustment(adj_list=adj_list,start=start,end=end,gains=gains_X, 
                                selected_element=node, spins=spins_X)
                X[size_X] = node
                size_X +=1 

            elif density_gain_Y >= obj_ratio_Y:
                obj_Y += gains_Y[node]
                gain_adjustment(adj_list=adj_list,start=start,end=end, 
                                gains=gains_Y, selected_element=node, spins=spins_Y)
                Y[size_Y] = node
                size_Y +=1 



    # print(size_X)
    # print(size_Y)

    # sprint(X[:10])
    # sprint(Y[:10])

    # print(obj_X)
    # print(obj_Y)

    # X_prime = np.zeros(N)
    X_prime=np.zeros(shape=(N,),dtype=np.float64)
    
    c_X = 0

    for idx in range(size_X-1,-1,-1):
        node = X[idx]
        # print(node)
        if  node_weights[node] + c_X <= budget:
            X_prime[node] = 1
            c_X += node_weights[node]

        else:
            break
    
    # print('c_X',c_X)

    Y_prime = np.zeros(shape=(N,),dtype=np.float64)
    
    c_Y = 0

    for idx in range(size_Y-1,-1,-1):
        node = Y[idx]
        # type(node)
        if  node_weights[node] + c_Y <= budget:
            Y_prime[node] = 1
            c_Y += node_weights[node]

        else:
            break

    # print('c_Y',c_Y)

    obj_X_prime = calculate_obj(adj_list=adj_list,start=start,end=end,solution=X_prime)
    obj_Y_prime = calculate_obj(adj_list=adj_list,start=start,end=end,solution=Y_prime)

    # obj_X_prime = int(obj_X_prime)

    # Determine which prime list to return based on the conditions
    if obj_X_prime >= max(obj_Y_prime, max_gain):
        return (obj_X_prime,X_prime)
    elif obj_Y_prime >= max_gain:
        return obj_Y_prime,Y_prime
    else:
        # return np.array([max_singleton])
        return max_gain,np.array([max_singleton],dtype=np.float64)
    
@njit
def calculate_l_prime(nodes,size, node_weights, bound):
    c = 0
    bound_size = 0
    
    # for node in nodes:
    for i in range(size):
        if c + node_weights[nodes[i]] <= bound:
            c += node_weights[nodes[i]]
            bound_size += 1
        else:
            break
    
    return bound_size, c
    

@njit
def get_max_obj(adj_list,start,end, nodes,size, node_weights,mask, budget, c_value):

    N = len(start)

    solution = np.zeros(N)

    for idx in range(size):
        solution[nodes[idx]] = 1

    
    # max_gain = 0 
    obj = calculate_obj(adj_list=adj_list,start=start,end=end,solution=solution)

    max_obj = obj

    max_gain_node = -1

    for node in range(N):
        if mask[node] and solution[node] == 0:
            gain = 0

            if node_weights[node] + c_value <= budget:

                for neighbor in adj_list[start[node]:end[node]]:
                    if solution[neighbor] == solution[node]:
                        gain += 1
                    else:
                        gain -= 1

            if max_obj<obj+gain:
                max_gain_node= node
                max_obj = max(max_obj,obj+gain) 
            # max_obj = max(max_obj,obj+gain) 

    solution[max_gain_node] = 1

    # print(max_obj)

    # assert max_obj == calculate_obj(adj_list=adj_list,start=start,end=end,solution=solution)
    
    return max_obj,solution
    
@njit 
def DLA_numba(adj_list,start,end,node_weights,budget,mask,eps=0.1):


    gains = get_gains(start,end)
    # S_prime = LA(adj_list=adj_list,start=start,end=end,
    #              gains=gains,node_weights=node_weights,budget=budget,mask=mask)
    
    
    
    # tau = calculate_obj(adj_list=adj_list,start=start,end=end,solution=S_prime)
    tau,best_solution = LA(adj_list=adj_list,start=start,end=end,
                 gains=gains,node_weights=node_weights,budget=budget,mask=mask)
    # print(tau)

    eps_prime = eps / 14
    delta = np.ceil(np.log(1/eps_prime)/eps_prime)
    theta = 19 * tau / (6*eps_prime*budget)


    gains = get_gains(start,end)

    N= len(start)
    X = np.zeros(shape=(N,),dtype=np.int32)
    Y = np.zeros(shape=(N,),dtype=np.int32)

    size_X = 0
    size_Y = 0

    gains_X = gains.copy()
    gains_Y = gains.copy()

    c_X = 0 
    c_Y = 0

    spins_X = np.zeros(N)
    spins_Y = np.zeros(N)

    while theta >= tau * (1 - eps_prime) / (6 * budget):

        for node in range(N):
            if mask[node] == 0 or spins_X[node] == 1  or spins_Y[node] == 1:
                continue
            
            # Calculate density gains
            density_gain_X = gains_X[node] / node_weights[node]
            density_gain_Y = gains_Y[node] / node_weights[node]
            max_density_gain = max(density_gain_X, density_gain_Y)

            if max_density_gain >= theta:
                # Calculate potential new weights
                new_weight_X = node_weights[node] + c_X
                new_weight_Y = node_weights[node] + c_Y

                # Check budget constraints for both X and Y
                if new_weight_X <= budget and new_weight_Y <= budget:
                    if density_gain_X >= density_gain_Y:
                        gain_adjustment(adj_list=adj_list,start=start,end=end, 
                                        gains=gains_X, selected_element=node, spins=spins_X)
                        # X[node] =1 
                        X[size_X] = node
                        size_X += 1 
                        c_X = new_weight_X
                    else:
                        gain_adjustment(adj_list=adj_list,start=start,end=end, 
                                        gains=gains_Y, selected_element=node, spins=spins_Y)
                        Y[size_Y] = node
                        size_Y += 1
                        c_Y = new_weight_Y

                elif new_weight_X <= budget:
                    gain_adjustment(adj_list=adj_list,start=start,end=end, 
                                    gains=gains_X, selected_element=node, spins=spins_X)
                    X[size_X] = node
                    size_X += 1
                    c_X = new_weight_X

                elif new_weight_Y <= budget:
                    gain_adjustment(adj_list=adj_list,start=start,end=end, 
                                    gains=gains_Y, selected_element=node, spins=spins_Y)
                    Y[size_Y] = node
                    size_Y += 1
                    c_Y = new_weight_Y


        theta *= (1 - eps_prime)


    # print(c_X)
    # print(c_Y)

    solution_X = np.zeros(N)

    for idx in range(size_X):
        solution_X[X[idx]] =1

    solution_Y = np.zeros(N)

    for idx in range(size_Y):
        solution_Y[Y[idx]] =1

   
    

    S = max(tau,
            calculate_obj(adj_list,start,end,solution_X),
            calculate_obj(adj_list,start,end,solution_Y))
    
    if S == calculate_obj(adj_list,start,end,solution_X):
        best_solution = solution_X.copy()
    
    elif S == calculate_obj(adj_list,start,end,solution_Y):
        best_solution = solution_Y.copy()
        

    
    for l in range(int(delta)):
        # sprint(l)

        bound = eps_prime*budget * (1+eps_prime)** l
        # print('Bound:',bound)

        bound_size_X,c_X = calculate_l_prime(X,size_X, node_weights, bound)
        bound_size_Y,c_Y = calculate_l_prime(Y,size_Y, node_weights, bound)
        # print(bound_size_X,c_X)
        # print(bound_size_Y,c_Y)

        # Get max objectives for X and Y
        obj_X,sol_X = get_max_obj(adj_list, start, end, X, bound_size_X, node_weights, mask, budget, c_X)
        obj_Y,sol_Y = get_max_obj(adj_list, start, end, Y, bound_size_Y, node_weights, mask, budget, c_Y)

        # Update the best objective and corresponding solution if a new maximum is found
        if obj_X > S:
            S = obj_X
            best_solution = sol_X
        if obj_Y > S:
            S = obj_Y
            best_solution = sol_Y
        # S = max(S,get_max_obj(adj_list,start,end,X,bound_size_X, node_weights,mask, budget, c_X))
        
        # S = max(S,get_max_obj(adj_list,start,end, Y,bound_size_Y, node_weights,mask, budget, c_Y))
        # print(S)
        
    # print('S',S)
    return S,best_solution
    

    

def DLA(graph,budget,node_weights,ground_set=None):

    adj_list,start,end = flatten_graph(graph=graph)
    N = len(start)

    node_weights = np.array([node_weights[i] for i in range(N)])

    

    if ground_set:
        mask = np.zeros(N)
        for node in ground_set:
            mask[node] = 1
    else:

        mask = np.ones(N)

    # DLA_numba(adj_list,start,end,node_weights,budget,mask,eps=0.1)
    objval,solution = DLA_numba(adj_list=adj_list,start=start,end = end,
             node_weights=node_weights,
             mask=mask,budget=budget)
    
    return objval,solution
    


    