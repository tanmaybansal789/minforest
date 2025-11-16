import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import minforest as mf

if __name__ == "__main__":
    iris = load_iris()
    print(iris.DESCR)
    x = iris.data.astype(np.float32)
    y = iris.target.astype(np.int32)


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    tree = mf.DecisionTree(x, y)

    print(tree.root)
    preds = tree.predict(x_test)
    print(np.mean(preds == y_test))