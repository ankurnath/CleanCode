from utils import *


def train_test_split (dataset:str,ratio:float,edge_level_split:bool,seed:int):

    np.random.seed(seed)
    random.seed(seed)
    log_file = f"{dataset}_train_test_split.log"

    
    logging.basicConfig(level=logging.INFO, filename=os.path.join(log_file),
                        filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')


    logging.info(f"Starting train_test_split with dataset={dataset}, ratio={ratio}, edge_level_split={edge_level_split}")
    # graph = nx.read_edgelist(f'../../data/snap_dataset/{dataset}.txt', create_using=nx.Graph(), nodetype=int)
    

    current_folder = os.getcwd()
    file_path = os.path.join(current_folder,f'data/snap_dataset/{dataset}.txt')
    graph = load_graph(file_path)

    # graph = load_graph(f'../../data/snap_dataset/{dataset}.txt')
    logging.info(f"Graph loaded with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    
    # train_save_folder='../../data/train'

    train_save_folder= os.path.join(current_folder,'data/train')
    test_save_folder= os.path.join(current_folder,'data/test')

    # test_save_folder='../../data/test'

    os.makedirs(train_save_folder,exist_ok=True)
    os.makedirs(test_save_folder,exist_ok=True)
    logging.info(f"Train and test directories created or already exist")
    train_file_path = os.path.join(train_save_folder,f'{dataset}')
    test_file_path = os.path.join(test_save_folder,f'{dataset}')



    if edge_level_split:
        
        edges = np.array(graph.edges())
        indices = [i for i in range(len(edges))]
        random.shuffle(indices)
        train_ids = edges[indices[:int(ratio * graph.number_of_edges())]]
        test_ids = edges[indices[int(ratio * graph.number_of_edges()):]]
        logging.info(f" Train edges: {len(train_ids)}, Test edges: {len(test_ids)}")
        train_graph = nx.Graph()
        train_graph.add_edges_from(train_ids)
        for node in list(train_graph.nodes()):
            if train_graph.degree(node) == 0:
                train_graph.remove_node(node)
        logging.info(f"Train graph created with {train_graph.number_of_nodes()} nodes and {train_graph.number_of_edges()} edges")
        save_to_pickle(data=train_graph,file_path=train_file_path)

        test_graph = nx.Graph()
        test_graph.add_edges_from(test_ids)
        for node in list(test_graph.nodes()):
            if test_graph.degree(node) == 0:
                test_graph.remove_node(node)
        logging.info(f"Test graph created with {test_graph.number_of_nodes()} nodes and {test_graph.number_of_edges()} edges")
        save_to_pickle(data=test_graph,file_path=test_file_path)

    else:
        raise NotImplementedError('Node level splitting not implemented')

        


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument( "--ratio",type=float,default=0.3, help="Train test ratio" )
    parser.add_argument( "--seed",type=int,default=0,help="Train test ratio" )
    parser.add_argument( "--edge_level_split",type=bool,default=True,help="Edge level split" )
    parser.add_argument( "--dataset", type=str,default='Facebook', required=True, help="Name of the dataset to be used (default: 'Facebook')" )

    args = parser.parse_args()

    train_test_split(dataset=args.dataset,ratio=args.ratio,seed=args.seed,edge_level_split=args.edge_level_split)
