import numpy as np


#####################
# Utility functions #
#####################

def entropy(y):
    # Get the different classes that exist in the output, and their counts
    classes, counts = np.unique(y, return_counts=True)
    # Turn in to probability distribution (sums to 1)
    prob = counts / counts.sum()
    # We want to penalise having an even distribution - we prefer the data to be mostly in a specific class
    # We calculate the average number of bits (log2) needed to describe the class of a random sample
    return -np.sum(prob * np.log2(prob))

#####################
# Helper structures #
#####################

class DecisionNode:
    """ Represents a single numerical decision, splitting data based on a specific feature and threshold. """

    def __init__(self, feature, threshold, left, right, value=None):
        self.feature = feature     # The index of the feature we split our data on
        self.threshold = threshold # If under threshold, go to left node, otherwise right node
        self.left = left           # The left node
        self.right = right         # The right node
        self.value = value         # If this is a leaf node, the predicted class

    @staticmethod
    def make_leaf(value):
        return DecisionNode(None, None, None, None, value)

    def is_leaf(self):
        return self.value is not None

######################
# Main class + logic # 
######################

class DecisionTree:
    """ Represents a tree of DecisionNodes """
    def __init__(self, x, y, max_depth=10):
        self.root = DecisionTree._build(x, y, 0, max_depth)

    @staticmethod
    def _best_split(x, y):
        n_samples, n_features = x.shape
        best = {
            'feature'   : None,
            'threshold' : None,
            'gain'      : -np.inf
        }
        old_entropy = entropy(y)

        for feature in range(n_features):
            values = np.sort(np.unique(x[:, feature]))
            # Average between adjacent values
            thresholds = (values[:-1] + values[1:]) / 2

            for t in thresholds:
                left_mask = x[:, feature] <= t
                right_mask = x[:, feature] > t

                # If none of the data is contained in one of the masks, then this is a useless threshold
                if not any(left_mask) or not any(right_mask):
                    continue
                    
                left_entropy = entropy(y[left_mask])
                right_entropy = entropy(y[right_mask])

                # .mean() on a boolean mask gets what fraction of the values are put into mask
                # We use it as our weight (left_mask.mean() + right_mask.mean())
                new_entropy = (left_mask.mean() * left_entropy + 
                               right_mask.mean() * right_entropy)

                # The gain is just the reduction in entropy
                gain = old_entropy - new_entropy
                if gain > best['gain']:
                    best['feature'] = feature
                    best['threshold'] = t
                    best['gain'] = gain
        
        return best

    @staticmethod
    def _build(x, y, depth, max_depth):
        # If we've gone to our maximum depth, or we've only got 1 remaining class
        classes, counts = np.unique(y)
        if depth >= max_depth or len(classes) == 1:
            # We use the highest count to choose our value at the end
            return DecisionNode.make_leaf(classes[np.argmax(counts)])
        
        best = DecisionTree._best_split(x, y)
        # No valid splits
        if best['gain'] == 0:
            return DecisionNode.make_leaf(classes[np.argmax(counts)])
            
        left_mask  = x[:, best['feature']] <= best['threshold']
        right_mask = x[:, best['feature']] >  best['threshold']

        left  = DecisionTree._build(x[left_mask],  y[left_mask],  depth + 1, max_depth)
        right = DecisionTree._build(x[right_mask], y[right_mask], depth + 1, max_depth)

        return DecisionNode(
            best['feature'],
            best['threshold'],
            left,
            right,
        )

    @staticmethod
    def _predict_helper(node, x):
        if node.is_leaf():
            return node.value
            
        return DecisionTree._predict_helper(
            node.left if x[node.feature] <= node.threshold else node.right,
            x
        )
    
    def predict(self, x):
        return np.array([DecisionTree._predict_helper(self.root, xr) for xr in x])
    
