## Setup

The included [``environment.yml``](environment.yml) file will produce a working environment called ``pruning``.



    >>> cd Pruning
    >>> conda env create -f environment.yml 
    >>> conda activate pruning


## Data Download

To download datasets, run the following command. Replace Facebook and Wiki with your desired dataset names.

    >>> python data_processing/data_download.py --datasets Facebook Wiki

To split the dataset into training and test sets, specify the dataset and the desired split ratio (e.g., 30% train data):

    >>> python data_processing/train_test_split.py --dataset Facebook --ratio 0.3 


## Remark 

For the implementation details of COMBHelper, LeNSE, and GCOMB-P, please refer to the original repository.


## MaxCover

The MaxCover problem can be solved under size or knapsack constraints using different algorithms. Below are instructions for using QuickPrune, GNNPruner, and SS methods with the MaxCover problem. 
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

