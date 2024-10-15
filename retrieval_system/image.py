from utils import *


from image_utils import *

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default= 'food101', help="Dataset")
    parser.add_argument('--max_budget', type = int ,default=100, help = 'Maximum Budget')
    parser.add_argument('--min_budget', type = int ,default=10, help = 'Minimum Budget')
    parser.add_argument("--delta", type=float, default=0.05, help="Delta")
    parser.add_argument("--eps", type=float, default=0.1, help="eps")
    parser.add_argument("--eta",type =float,default=0.5,help="Eta")

    seed = 42
    
    args = parser.parse_args()

    dataset_name = args.dataset
    max_budget =args.max_budget
    min_budget = args.min_budget
    eta = args.eta
    delta = args.delta 
    eps = args.eps

    print('dataset',dataset_name )

    # load dataset
    load_paths = {'food101':"nateraw/food101",
                  'beans':"beans",
                  'cifar100':"uoft-cs/cifar100"
                  }
    model_ckpts = {'food101':'nateraw/vit-base-food101',
                   'beans':"nateraw/vit-base-beans",
                   'cifar100':'Ahmed9275/Vit-Cifar100'}

    dataset = load_dataset(load_paths[dataset_name])

    if dataset_name =='beans':
        dataset = dataset.rename_columns({'labels':'label'})

    elif dataset_name == 'cifar100':
        dataset = dataset.rename_columns({'img':'image','fine_label':'label'})

    if dataset_name == 'food101':
        samples = []
        max_count = 100

        # # Create a dictionary to keep count of samples per label
        label_counts = defaultdict(int)

        # # Select samples
        for idx, label in tqdm(enumerate(dataset["train"]['label'])):

            # Only add sample if count for this label is less than max_count
            if label_counts[label] < max_count:
                samples.append(idx)
                label_counts[label] += 1
        candidate_subset = dataset["train"].select(samples).shuffle(seed=seed)

    elif dataset_name == 'beans':
        skewed_samples =[]
        max_count = 10
        cnt =0
        for idx,label in enumerate(dataset["train"]['label']):
            if label ==0:
                if cnt<max_count:
                    skewed_samples.append(idx)
                    cnt+=1

            else:
                skewed_samples.append(idx)
        candidate_subset = dataset["train"].select(skewed_samples).shuffle(seed=seed)


    elif dataset_name == 'cifar100':
        candidate_subset = dataset["train"].shuffle(seed=seed)

    else:
        raise ValueError('Unknownd dataset')


    model_ckpt = model_ckpts[dataset_name]
    extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
    model = AutoModel.from_pretrained(model_ckpt)
    hidden_dim = model.config.hidden_size

    transformation_chain = T.Compose(
        [
            # We first resize the input image to 256x256 and then we take center crop.
            T.Resize(int((256 / 224) * extractor.size["height"])),
            T.CenterCrop(extractor.size["height"]),
            T.ToTensor(),
            T.Normalize(mean=extractor.image_mean, std=extractor.image_std),
        ]
    )

    batch_size = 24
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extract_fn = extract_embeddings(model.to(device),transformation_chain)
    candidate_subset_emb = candidate_subset.map(extract_fn, batched=True, batch_size=256)


    all_candidate_embeddings = np.array(candidate_subset_emb["embeddings"])
    all_candidate_embeddings = torch.from_numpy(all_candidate_embeddings)

    candidate_similarity = compute_scores(all_candidate_embeddings,all_candidate_embeddings)

    # Seed for reproducibility
    random.seed(42)
    # Labels you want to sample from (e.g., labels 0-9)
    labels_to_sample = list(range(101))


    # Dictionary to keep track of samples per label
    label_samples = {label: [] for label in labels_to_sample}

    # if dataset_name == 'beans':
    #     label_feature = 'labels'
    # elif dataset_name == 'food101':
    #     label_feature = 'label'

    # Loop over dataset and group indices by label

    if dataset_name == 'cifar100':
        sample_dist = 'test'
    else:
        sample_dist = "validation"
    

    for idx, label in enumerate(dataset[sample_dist]['label']):
        if label in labels_to_sample:
            label_samples[label].append(idx)


    for label in [0]:
        query_indices = []
        sampled_indices = random.sample(label_samples[label],5)
        query_indices.extend(sampled_indices)

        # Select the images using the sampled indices
        query_images = [image if isinstance(image, Image.Image) else Image.open(image).convert("RGB") 
                        for image in dataset[sample_dist].select(query_indices)["image"]]

    
        query_images_transformed_batch = torch.stack([transformation_chain(query_image) for query_image in query_images])
        query_images_transformed_batch = query_images_transformed_batch.to(device)
        # Prepare the batch dictionary
        new_batch = {"pixel_values": query_images_transformed_batch}
        # Compute the embeddings for all images in the batch
        with torch.no_grad():
            query_embeddings = model(**new_batch).last_hidden_state[:, 0].cpu()




        query_similarity = compute_scores(query_embeddings,all_candidate_embeddings)
       
        df = defaultdict(list)
        delta = 0.01

        ground_set_SS = SS(candidate_similarity,query_similarity)

        size_SS = round(np.sum(ground_set_SS)/len(ground_set_SS)*100,4)
        for budget in [5,10,15,20]:

            ### SS
            obj_val_CUT_SS,solution_CUT_SS,_=graph_cut(candidate_similarity=candidate_similarity,
                                    query_similarity=query_similarity,
                                    budget=budget,ground_set=ground_set_SS)


            ########

            N = len(candidate_similarity)

            

            ground_set_QS = QS(candidate_similarity,
                               query_similarity,
                               max_budget=max_budget,
                               min_budget=min_budget,
                               delta=delta,
                               eta=eta,
                               eps=eps)
            
            size_QS = round(np.sum(ground_set_QS)/len(ground_set_QS)*100,4)
            

            obj_val_CUT_QS,solution_CUT_QS,_=graph_cut(candidate_similarity=candidate_similarity,
                                    query_similarity=query_similarity,
                                    budget=budget,ground_set=ground_set_QS)
            
            

            

            obj_val_CUT,solution_CUT,_=graph_cut(candidate_similarity=candidate_similarity,
                                    query_similarity=query_similarity,
                                    budget=budget,ground_set=np.ones(N))
            
            num_repeat = 5
            obj_val_CUT_Random = 0
            for i in range(num_repeat):
                # Randomly select a mask
                random_mask = np.random.choice(np.arange(N), size=int(ground_set_QS.sum()), replace=False)
                
                # Create the random ground set
                random_ground_set = np.zeros(N)
                random_ground_set[random_mask] = 1 
                
                # Calculate the objective value using graph_cut
                temp_obj_val_CUT_Random,solution_CUT_Random,_= graph_cut(candidate_similarity=candidate_similarity,
                                            query_similarity=query_similarity,
                                            budget=budget,
                                            ground_set=random_ground_set)
                
                obj_val_CUT_Random+=temp_obj_val_CUT_Random

                
            
            obj_val_CUT_Random /= num_repeat


            df['Budget'].append(budget)
            df['Ratio_Obj(QS)'].append(obj_val_CUT_QS/obj_val_CUT)
            df['Pg(QS)'].append(size_QS)
            df['Ratio_Obj(SS)'].append(obj_val_CUT_SS/obj_val_CUT)
            df['Pg(SS)'].append(size_SS)
            df['Obj'].append(obj_val_CUT)
            df['Ratio_Obj(Random)'].append(obj_val_CUT_Random/obj_val_CUT)
            
        df = pd.DataFrame(df)
        print(df)
        current_folder = os.getcwd()
        save_folder = os.path.join(current_folder,'retrieval_system',f'data/{dataset_name}')
        os.makedirs(save_folder,exist_ok=True)
        # save_path = os.path.join(current_folder,'retrieval_system',f'data/image_{args.dataset}')
        save_path = os.path.join(save_folder,f'image_{dataset_name}')
        df.to_pickle(save_path)



