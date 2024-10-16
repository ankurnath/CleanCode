## Setup

The included [``environment.yml``](environment.yml) file will produce a working environment called ``pruning``.

    >>> cd Pruning
    >>> conda env create -f environment.yml 
    >>> conda activate pruning


## Data Download

For downloading the data

    >>> python data_processing/data_download.py --datasets Facebook Wiki

For train and test split 

    >>> python data_processing/train_test_split.py --dataset Facebook --ratio 0.3 

## MaxCover

For size-constraint with quickprune
    >>> python MaxCover/size_quickprune.py --dataset Facebook 

For size-constraint with GNNpruner
    >>> python MaxCover/size_gnnpruner.py --dataset Facebook
    
For size-constraint with SS
    >>> python MaxCover/SS.py --dataset Facebook

For knapsack-constraint with quickprune
    >>> python MaxCover/knapsack_quickprune.py --dataset Facebook 

For knapsack-constraint with GNNpruner
    >>> python MaxCover/knapsack_gnnpruner.py --dataset Facebook

