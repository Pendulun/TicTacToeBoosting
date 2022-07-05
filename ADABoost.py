class ADABoost():
    
    def __init__(self, num_trees, tree_h):
        self._max_n_trees = num_trees
        self._max_tree_height = tree_h
    
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
    
    def fit(self, dataframe, target_col):
        pass

    def predict(self, dataframe, target_col) -> list:
        predictions = [0] * len(dataframe)

        return predictions