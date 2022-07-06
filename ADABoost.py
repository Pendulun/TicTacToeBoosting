from sklearn import tree
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
        while tree_count < self._max_n_trees:
            curr_tree = tree.DecisionTreeClassifier(max_depth=self._max_tree_height, random_state=self._random_seed)
            curr_tree.fit(x, y, sample_weight=sample_weights)

            #Pegar onde essa árvore errou

            #Atualizar os pesos

            tree_count += 1
            self._trees.append(curr_tree)
            self._trees_alphas.append(1)
    
    def _get_starting_weights(self, data_len:int) ->np.array:
        return np.full((data_len), 1/data_len)

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