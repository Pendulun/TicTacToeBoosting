from sklearn import tree
import pandas as pd

class ADABoost():
    
    def __init__(self, num_trees:int, tree_h:int, random_seed:int):
        self._max_n_trees = num_trees
        self._max_tree_height = tree_h
        self._random_seed = random_seed
        self._tree = None
    
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
        my_tree = tree.DecisionTreeClassifier(max_depth=self._max_tree_height, random_state=self._random_seed)
        
        x = dataframe.drop(target_col, axis=1)
        y = dataframe[target_col]
        my_tree.fit(x, y)

        self._tree = my_tree

    def predict(self, dataframe:pd.DataFrame, target_col:str) -> list:
        x = dataframe.drop(target_col, axis=1)
        return self._tree.predict(x)