import wget
import os
import gzip
from argparse import ArgumentParser



def download(datasets):

    DATASETS_URLS={ 
                "Facebook":"https://snap.stanford.edu/data/facebook_combined.txt.gz",
                "web-Google":"https://snap.stanford.edu/data/web-Google.txt.gz",
                "Deezer" : "https://snap.stanford.edu/data/gemsec_deezer_dataset.tar.gz",
                "Wiki": "https://snap.stanford.edu/data/wiki-Vote.txt.gz",
                "DBLP":"https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz",
                "Slashdot": "https://snap.stanford.edu/data/soc-sign-Slashdot081106.txt.gz",
                "Skitter":"https://snap.stanford.edu/data/as-skitter.txt.gz",
                "YouTube":"https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz",
                "Twitter":"https://snap.stanford.edu/data/twitter_combined.txt.gz",
                }

    current_folder = os.getcwd()
    print(current_folder)
    save_path= os.path.join(current_folder,f'data/snap_dataset')
    os.makedirs(save_path,exist_ok=True)


    # Download datasets
    # for dataset_name, dataset_url in DATASETS_URLS.items():
    for dataset_name in datasets:
        if dataset_name not in DATASETS_URLS:
            print('URL not found')
            continue
        dataset_url=  DATASETS_URLS[dataset_name]
        dataset_file_path = os.path.join(save_path, f"{dataset_name}.txt.gz")

        # Check if the dataset file already exists before downloading
        if not os.path.exists(dataset_file_path):
            print(f"Downloading {dataset_name} dataset...")
            wget.download(dataset_url, dataset_file_path)
            print(f"{dataset_name} downloaded successfully.")
        else:
            print(f"{dataset_name} has already been downloaded. Skipping.")


    for dataset in datasets:
        input_file =  os.path.join(save_path,f'{dataset}.txt.gz')
        output_file = os.path.join(save_path,f'{dataset}.txt')

        # Check if the output file already exists before decompressing
        if not os.path.exists(output_file):
            print(f"Decompressing {dataset} from {input_file} to {output_file}...")
            with gzip.open(input_file, 'rb') as f_in, open(output_file, 'wb') as f_out:
                f_out.writelines(f_in)
            print(f"{dataset} decompressed successfully.")
        else:
            print(f"{dataset} has already been decompressed. Skipping.")

        # os.remove(input_file)

        



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument( "--datasets", type=str,nargs='+', default='Facebook', help="Name of the dataset to be used (default: 'Facebook')" )
    
    args = parser.parse_args()

    download(datasets=args.datasets)