import argparse
import sys

# https://stackoverflow.com/a/12117089/16264901
class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end
    
    def __repr__(self):
        return '[{0}, {1}]'.format(self.start, self.end)

class MinRange(object):
    def __init__(self, start):
        self.start = start

    def __eq__(self, other):
        return self.start <= other
    
    def __repr__(self):
        return '[start:{0}]'.format(self.start)

def get_arg_parser():
    my_parser = argparse.ArgumentParser(description='Process args.')

    my_parser.add_argument('--data_path', type=str, help="The dataset file path. Required",
                            metavar="path", required=True
                        )
    
    DEFAULT_TRAIN_AMOUNT = 0.7
    MIN_TRAIN_AMOUNT = 0.0
    MAX_TRAIN_AMOUNT = 1.0

    train_help_msg = "The percentage of train data."
    train_help_msg += f" Must be a float [{MIN_TRAIN_AMOUNT}, {MAX_TRAIN_AMOUNT}]."
    train_help_msg += f" Default={DEFAULT_TRAIN_AMOUNT}"
    my_parser.add_argument('--train_split', type= float, metavar= "train_amount",
                            choices= [Range(MIN_TRAIN_AMOUNT, MAX_TRAIN_AMOUNT)], 
                            default= DEFAULT_TRAIN_AMOUNT,
                            help= train_help_msg
                        )
    
    DEFAULT_N_TREES = 10
    MIN_N_TREES = 1
    n_trees_help_msg = f"The number of trees to use in the ADABoost."
    n_trees_help_msg += f" Minimum: {MIN_N_TREES}."
    n_trees_help_msg += f" Default = {DEFAULT_N_TREES}"

    my_parser.add_argument("--n_trees", metavar="n_trees",
                            type=int, default=DEFAULT_N_TREES, choices=[MinRange(MIN_N_TREES)],
                            help=n_trees_help_msg
                        )
    
    DEFAULT_TREE_HEIGHT = 1
    MIN_TREE_HEIGHT = 1
    tree_height_help_msg = f"The max tree height to use."
    tree_height_help_msg += f" Minimum: {MIN_TREE_HEIGHT}."
    tree_height_help_msg += f" Default = {DEFAULT_TREE_HEIGHT}"
    my_parser.add_argument("--tree_height", metavar="height",
                            type=int, default=DEFAULT_TREE_HEIGHT, choices=[MinRange(MIN_TREE_HEIGHT)],
                            help=tree_height_help_msg
                        )

    return my_parser

if __name__ == "__main__":
    my_parser = get_arg_parser()
    my_parser.parse_args()