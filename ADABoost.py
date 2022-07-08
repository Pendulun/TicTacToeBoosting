from math import exp, log
from sklearn import tree
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

class ADABoost():
    
    def __init__(self, num_trees:int, tree_h:int, random_seed:int):
        self._max_n_trees = num_trees
        self._max_tree_height = tree_h
        self._random_seed = random_seed
        self._trees = list()
        self._trees_alphas = list()
    
    @property
    def max_n_trees(self):
        """
        The maximum amount of trees used
        """
        return self._max_n_trees
    
    @max_n_trees.setter
    def max_n_trees(self, new_max):
        if type(new_max) != int:
            raise TypeError("max_n_trees must be an int!")
        
        if new_max < 1:
            raise ValueError("max_n_trees must be a positive int!")
        
        self._max_n_trees = new_max
    
    @property
    def max_tree_height(self):
        """
        The maximum height of the trees
        """
        return self._max_tree_height
    
    @max_tree_height.setter
    def max_tree_height(self, new_max):
        if type(new_max) != int:
            raise TypeError("max_tree_height must be an int!")
        
        if new_max < 1:
            raise ValueError("max_tree_height must be a positive int!")
        
        self._max_tree_height = new_max
    
    def fit(self, dataframe:pd.DataFrame, target_col:str):
        sample_weights = self._get_starting_weights(len(dataframe))

        x = dataframe.drop(target_col, axis=1)
        y = dataframe[target_col]

        tree_count = 0
        curr_it = 1
        while tree_count < self._max_n_trees:

            curr_tree = self._get_new_tree()
            curr_tree.fit(x, y, sample_weight=sample_weights)

            curr_tree_pred = curr_tree.predict(x)
            
            curr_tree_error = self._get_tree_error(sample_weights, y.values, curr_tree_pred)
            curr_tree_alpha = self._get_tree_alpha( curr_it, curr_tree_error)

            sample_weights = self._update_weights(sample_weights, y.values, curr_tree_pred, curr_tree_alpha)
            
            self._trees.append(curr_tree)
            self._trees_alphas.append(curr_tree_alpha)

            if curr_tree_error == 0:
                break

            tree_count += 1
            curr_it += 1
    
    def _get_starting_weights(self, data_len:int) ->np.ndarray:
        return np.full((data_len), 1/data_len)

    def _get_new_tree(self):
        return tree.DecisionTreeClassifier(max_depth=self._max_tree_height, random_state=self._random_seed)

    def _get_tree_error(self, sample_weights, true_y:np.ndarray, pred_y:np.ndarray) -> float:
        curr_tree_accuracy = accuracy_score(true_y, pred_y, normalize=True, sample_weight=sample_weights)
        print(f"Curr Tree acc: {curr_tree_accuracy}")
        curr_tree_error = 1-curr_tree_accuracy
        return curr_tree_error

    def _get_tree_alpha(self, curr_it:int, error:float) -> float:
        #default value if error is 0
        it_weighted_error = 1/curr_it
        if error != 0:
            it_weighted_error = (error)**curr_it
        else:
            print(f"error == 0 at it {curr_it}")
        
        curr_tree_alpha = 0.5
        if it_weighted_error != 1:
            curr_tree_alpha = 0.5*(log((1-(it_weighted_error))/it_weighted_error))
        else:
            print("it_weighted_error == 1")

        return curr_tree_alpha
    
    def _update_weights(self, sample_weights:np.ndarray, true_y:pd.DataFrame, pred_y:np.ndarray, alpha:float) ->np.ndarray:
        input_size = len(true_y)
        
        euler_vector = np.fromiter(self._euler_exponents(alpha, pred_y, true_y), dtype=float, count = input_size)

        sample_weights = sample_weights*euler_vector
        sample_weights /= np.sum(sample_weights)
        return sample_weights
    
    def _euler_exponents(self, alpha:float, pred_y:np.ndarray, true_y:np.ndarray):
        for label_idx in range(true_y.shape[0]):
            yield exp(-1*alpha*pred_y[label_idx]*true_y[label_idx])

    def predict(self, dataframe:pd.DataFrame, target_col:str) -> list:
        x = dataframe.drop(target_col, axis=1)
        return self._predict_from_trees(x)
    
    def _predict_from_trees(self, x:pd.DataFrame) -> int:
        
        results = [tree.predict(x) for tree in self._trees]

        final_results = list()
        num_inputs = len(results[0])
        for curr_input_prediction_idx in range(num_inputs):
            predictions_for_curr_input = [preds[curr_input_prediction_idx] for preds in results]
            final_results.append(self._weighted_most_frequent(predictions_for_curr_input))

        return final_results
    
    def _weighted_most_frequent(self, numbers:list) -> float:

        weight_per_result = dict()

        biggest_weight = -1
        biggest_weight_result = None
        for result_idx in range(len(numbers)):
            curr_result = numbers[result_idx]
            weight_per_result[curr_result] = weight_per_result.get(curr_result, 0) + self._trees_alphas[result_idx]
            
            curr_result_weight = weight_per_result[curr_result]
            if curr_result_weight > biggest_weight:
                biggest_weight = curr_result_weight
                biggest_weight_result = curr_result

        return biggest_weight_result
    
    def get_accuracy(self, true_y, pred_y) -> float:
        return accuracy_score(true_y, pred_y)