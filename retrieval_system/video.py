from utils import *
# video_utils.py
from video_utils import *
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default= 'ucf101', help="Dataset")
    parser.add_argument('--max_budget', type = int ,default=100, help = 'Maximum Budget')
    parser.add_argument('--min_budget', type = int ,default=10, help = 'Minimum Budget')
    parser.add_argument("--delta", type=float, default=0.01, help="Delta")
    parser.add_argument("--eps", type=float, default=0.1, help="eps")
    parser.add_argument("--eta",type =float,default=0.5,help="Eta")
    args = parser.parse_args()

    dataset_name = args.dataset
    max_budget =args.max_budget
    min_budget = args.min_budget
    eta = args.eta
    delta = args.delta 
    eps = args.eps

    print('dataset',dataset_name )

    embeddings=torch.load(f'data/{dataset_name}/all_embeddings.pt',weights_only=False)
    annotations = load_from_pickle(f'data/{dataset_name}/annotations')
    # print(annotations)
    annotations_list =list(annotations.values())
    embeddings_with_labels =[[annotations_list[idx],embeddings[idx]] 
                         for idx in range(len(annotations_list))]
    
    random.shuffle(embeddings_with_labels)

    all_candidate_embeddings_with_labels = embeddings_with_labels[:6000]
    all_query_embeddings_with_labels = embeddings_with_labels[6000:]

    all_candidate_embeddings= [item[1] for item in all_candidate_embeddings_with_labels]
    all_query_embeddings = [item[1] for item in all_query_embeddings_with_labels]

    all_candidate_labels= np.array([item[0] for item in all_candidate_embeddings_with_labels])
    all_query_labels = np.array([item[0] for item in all_query_embeddings_with_labels])
    all_candidate_embeddings = torch.stack(all_candidate_embeddings,dim=0)
    all_query_embeddings = torch.stack(all_query_embeddings,dim=0)


    queries_indices = [100,0,1]

    query_embeddings=all_query_embeddings[[queries_indices]]

    candidate_similarity = compute_scores(all_candidate_embeddings,all_candidate_embeddings)
    query_similarity = compute_scores(query_embeddings,all_candidate_embeddings)


    df = defaultdict(list)
delta = 0.001



costs = load_from_pickle(f'data/{dataset_name}/costs')




for budget in [5,10,15,20]:

    N = len(candidate_similarity)

    ground_set_QS = QS(candidate_similarity=candidate_similarity,
                       query_similarity=query_similarity,
                       costs = costs,
                       min_budget=min_budget,
                       max_budget=max_budget,
                       delta = delta,
                       eta = eta,
                       eps = eps)
    # size_SS = round(np.sum(ground_set_SS)/len(ground_set_SS)*100,4)
    size_QS = round(np.sum(ground_set_QS)/len(ground_set_QS)*100,4)
    retrived_images = np.where(ground_set_QS==1)[0]

    ### top-k

    gains = calculate_gains(candidate_similarity=candidate_similarity,
                            query_similarity=query_similarity,
                            solution_dense=np.zeros(N, dtype=np.int32),
                            size_solution=0)
    
    top_k_elements = np.argsort(gains)[::-1][:int(np.sum(ground_set_QS))]

    ground_set_top_k = np.zeros(N)
    for element in top_k_elements:
        ground_set_top_k[int(element)] = 1

    obj_val_top_k,solution_top_k,_=graph_cut(candidate_similarity=candidate_similarity,
                            query_similarity=query_similarity,costs=costs,
                            budget=budget,ground_set=ground_set_top_k)
    


    obj_val_CUT_QS,solution_CUT_QS,_=graph_cut(candidate_similarity=candidate_similarity,
                            query_similarity=query_similarity,costs=costs,
                            budget=budget,ground_set=ground_set_QS)
    
    

    

    obj_val_CUT,solution_CUT,_=graph_cut(candidate_similarity=candidate_similarity,
                            query_similarity=query_similarity,costs=costs,
                            budget=budget,ground_set=np.ones(N))
    
    num_repeat = 5
    obj_val_CUT_Random = 0
    for i in range(num_repeat):
        # Randomly select a mask
        random_mask = np.random.choice(np.arange(N), 
                                       size=int(ground_set_QS.sum()), replace=False)
        
        # Create the random ground set
        random_ground_set = np.zeros(N)
        random_ground_set[random_mask] = 1 
        
        # Calculate the objective value using graph_cut
        temp_obj_val_CUT_Random,solution_CUT_Random,_= graph_cut(candidate_similarity=candidate_similarity,
                                    query_similarity=query_similarity,
                                    budget=budget,costs=costs,
                                    ground_set=random_ground_set)
        
        obj_val_CUT_Random+=temp_obj_val_CUT_Random

        
    
    obj_val_CUT_Random /= num_repeat

    
    df['Budget'].append(budget)
    df['Ratio_Obj(QS)'].append(obj_val_CUT_QS/obj_val_CUT)
    df['Ratio_Obj(top-k)'].append(obj_val_top_k/obj_val_CUT)
    # df['Ratio_Obj(CUT+QS)'].append(obj_val_CUT_QS)
    df['Pg(QS)'].append(size_QS)
    df['Obj(CUT)'].append(obj_val_CUT)
    df['Ratio_Obj(Random)'].append(obj_val_CUT_Random/obj_val_CUT)

df = pd.DataFrame(df)

print(df)

current_folder = os.getcwd()
save_folder = os.path.join(current_folder,'retrieval_system',f'data/{dataset_name}')
os.makedirs(save_folder,exist_ok=True)

save_path = os.path.join(save_folder,f'video_{dataset_name}')
df.to_pickle(save_path)