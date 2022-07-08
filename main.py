import argparse
import pathlib
import pandas as pd
from ADABoost import ADABoost

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
    
    DEFAULT_RANDOM_SEED = 42
    MIN_RAND_SEED = 1
    rand_seed_help_msg = "The random seed to be used along the program."
    rand_seed_help_msg += f" Default: {DEFAULT_RANDOM_SEED}"
    my_parser.add_argument("--random_seed", metavar="seed",
                            type=int, default = DEFAULT_RANDOM_SEED, choices=[MinRange(MIN_RAND_SEED)],
                            help=rand_seed_help_msg
    )  

    return my_parser

def get_pos_mapping_dict():
    mapping_dict = dict()
    mapping_dict['x'] = 1
    mapping_dict['o'] = -1
    mapping_dict['b'] = 0
    
    mapping_dict['positive'] = 1
    mapping_dict['negative'] = -1

    return mapping_dict

def treat_data(data_df:pd.DataFrame) ->pd.DataFrame:
    pos_mapping_dict = get_pos_mapping_dict()
    data_df = data_df.applymap(lambda x: pos_mapping_dict[x])

    return data_df

def predict(my_args):

    data_df = pd.read_csv(my_args.data_path)
    TARGET_COL = 'x-win'

    data_df = treat_data(data_df)

    DEFAULT_RANDOM_SEED = 42
    
    data_df_train = data_df.sample(frac = my_args.train_split, random_state=DEFAULT_RANDOM_SEED, axis=0)
    data_df_test = data_df[~data_df.index.isin(data_df_train.index)]

    my_ada_boost = ADABoost(num_trees=my_args.n_trees, tree_h = my_args.tree_height, random_seed = my_args.random_seed)

    
    my_ada_boost.fit(data_df_train, target_col=TARGET_COL)
    predictions = my_ada_boost.predict(data_df_test, target_col=TARGET_COL)

    print(f"Final Predictions:\n{predictions}")
    print(f"Accuracy: {my_ada_boost.get_accuracy(data_df_test[TARGET_COL].values, predictions)}")

def check_data_file(data_file_path):
    data_file = pathlib.Path(data_file_path)
    if not data_file.exists():
        print(f"{data_file} does not exists!")
        exit(-1)

if __name__ == "__main__":
    my_parser = get_arg_parser()

    #Already gets from sys.argv
    my_args = my_parser.parse_args()

    check_data_file(my_args.data_path)
    
    predict(my_args)