"""DecisionTree - a single CART decision tree."""
from ._common import MLModel


class DecisionTree(MLModel):
    name, description = "DecisionTree", "Single CART decision tree"

    def _fit(self, X, y, w, Xv, yv, wv, seed, k):
        from sklearn.tree import DecisionTreeClassifier
        self.m = DecisionTreeClassifier(max_depth=24, min_samples_leaf=2, random_state=seed).fit(X, y, sample_weight=w)
