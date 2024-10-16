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

## MaxCut

Below are instructions for using QuickPrune, GNNPruner, and SS methods with the MaxCut problem. 
For size-constraint with quickprune

    >>> python MaxCut/size_quickprune.py --dataset Facebook 

For size-constraint with GNNpruner

    >>> python MaxCut/size_gnnpruner.py --dataset Facebook
    
For size-constraint with SS

    >>> python MaxCut/SS.py --dataset Facebook

For knapsack-constraint with quickprune

    >>> python MaxCut/knapsack_quickprune.py --dataset Facebook 

For knapsack-constraint with GNNpruner

    >>> python MaxCut/knapsack_gnnpruner.py --dataset Facebook


## Influence Maximization

Below are instructions for using QuickPrune, GNNPruner, and SS methods with the IM problem. 
For size-constraint with quickprune

    >>> python IM/size_quickprune.py --dataset Facebook 

For size-constraint with GNNpruner

    >>> python IM/size_gnnpruner.py --dataset Facebook
    
For size-constraint with SS

    >>> python IM/SS.py --dataset Facebook

For knapsack-constraint with quickprune

    >>> python IM/knapsack_quickprune.py --dataset Facebook 

For knapsack-constraint with GNNpruner

    >>> python IM/knapsack_gnnpruner.py --dataset Facebook


# Retrieval system

Image retrieval 

    >>> python retrieval_system/image.py --dataset beans


Video retrieval

    >>> python retrieval_system/video.py --dataset ucf101
