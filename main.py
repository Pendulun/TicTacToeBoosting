import argparse
import pathlib
import sys
import pandas as pd
from sklearn.model_selection import KFold
from ADABoost import ADABoost

# https://stackoverflow.com/a/12117089/16264901
class Range():
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end
  
    def __repr__(self):
        return '[{0}, {1}]'.format(self.start, self.end)

class MinRange():
    def __init__(self, start):
        self.start = start

    def __eq__(self, other):
        return self.start <= other

    def __repr__(self):
        return '[start:{0}]'.format(self.start)

def get_arg_parser():
    args_parser = argparse.ArgumentParser(description='Process args.')

    args_parser.add_argument('--data_path', type=str, help="The dataset file path. Required",
                            metavar="path", required=True
                        )

    DEFAULT_TRAIN_AMOUNT = 0.7
    MIN_TRAIN_AMOUNT = 0.0
    MAX_TRAIN_AMOUNT = 1.0

    train_help_msg = "The percentage of train data."
    train_help_msg += f" Must be a float [{MIN_TRAIN_AMOUNT}, {MAX_TRAIN_AMOUNT}]."
    train_help_msg += f" Default={DEFAULT_TRAIN_AMOUNT}"
    args_parser.add_argument('--train_split', type= float, metavar= "train_amount",
                            choices= [Range(MIN_TRAIN_AMOUNT, MAX_TRAIN_AMOUNT)], 
                            default= DEFAULT_TRAIN_AMOUNT,
                            help= train_help_msg
                        )
    
    DEFAULT_N_TREES = 10
    MIN_N_TREES = 1
    n_trees_help_msg = "The number of trees to use in the ADABoost."
    n_trees_help_msg += f" Minimum: {MIN_N_TREES}."
    n_trees_help_msg += f" Default = {DEFAULT_N_TREES}"

    args_parser.add_argument("--n_trees", metavar="n_trees",
                            type=int, default=DEFAULT_N_TREES, choices=[MinRange(MIN_N_TREES)],
                            help=n_trees_help_msg
                        )
    
    DEFAULT_RANDOM_SEED = 42
    MIN_RAND_SEED = 1
    rand_seed_help_msg = "The random seed to be used along the program."
    rand_seed_help_msg += f" Default: {DEFAULT_RANDOM_SEED}"
    args_parser.add_argument("--random_seed", metavar="seed",
                            type=int, default = DEFAULT_RANDOM_SEED, choices=[MinRange(MIN_RAND_SEED)],
                            help=rand_seed_help_msg
    )  

    return args_parser

def get_pos_mapping_dict():
    mapping_dict = dict()
    mapping_dict['x'] = 1
    mapping_dict['o'] = 0
    mapping_dict['b'] = -1
    
    mapping_dict['positive'] = 1
    mapping_dict['negative'] = -1

    return mapping_dict

def treat_data(data_df:pd.DataFrame) ->pd.DataFrame:
    pos_mapping_dict = get_pos_mapping_dict()
    data_df = data_df.applymap(lambda x: pos_mapping_dict[x])

    return data_df

def train_test_split(train_split, random_seed, data_df:pd.DataFrame) -> tuple:
    data_df_train = data_df.sample(frac = train_split, random_state=random_seed,
                                    axis=0)
    
    data_df_test = data_df[~data_df.index.isin(data_df_train.index)]
    return data_df_train,data_df_test

def get_kfold_mean_score(n_trees, rand_seed, y_train, x_train, n_folds=5) -> float:
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    scores = []
    kf = KFold(n_splits=n_folds)
    for train_index, test_index in kf.split(x_train):
        curr_model = ADABoost(n_trees, rand_seed)
        X_train_kf, X_test_kf = x_train[train_index], x_train[test_index]
        y_train_kf, y_test_kf = y_train[train_index], y_train[test_index]

        curr_model.fit(X_train_kf, y_train_kf)
        predictions = curr_model.predict(X_test_kf)
        scores.append(curr_model.get_accuracy(y_test_kf, predictions))
    
    return sum(scores)/len(scores)

def get_train_test_data(data_path:str, train_split:float, rand_seed:int):
    data_df = pd.read_csv(data_path)
    data_df = treat_data(data_df)
    
    data_df_train, data_df_test = train_test_split(train_split, rand_seed, data_df)

    TARGET_COL = 'x-win'
    y_train = data_df_train[TARGET_COL]
    x_train = data_df_train.drop(TARGET_COL, axis=1)

    y_test = data_df_test[TARGET_COL]
    x_test = data_df_test.drop(TARGET_COL, axis=1)
    return y_train,x_train,y_test,x_test

def predict(parsed_args):

    y_train, x_train, y_test, x_test = get_train_test_data(parsed_args.data_path, parsed_args.train_split, parsed_args.random_seed)

    kfold_mean_score = get_kfold_mean_score(parsed_args.n_trees, parsed_args.random_seed, y_train.values, x_train.values)
    print(f"k fold mean accuracy score: {kfold_mean_score}")

    final_model = ADABoost(parsed_args.n_trees, parsed_args.random_seed)
    final_model.fit(x_train.values, y_train.values)
    final_predictions = final_model.predict(x_test.values)

    test_acc = final_model.get_accuracy(y_test, final_predictions)
    print(f"Test accuracy: {test_acc}")

def check_data_file(data_file_path):
    data_file = pathlib.Path(data_file_path)
    if not data_file.exists():
        print(f"{data_file} does not exists!")
        sys.exit(-1)

if __name__ == "__main__":
    my_parser = get_arg_parser()

    #Already gets from sys.argv
    my_args = my_parser.parse_args()

    check_data_file(my_args.data_path)
    
    predict(my_args)
