"""RandomForest - bagged decision trees."""
from ... import settings as S
from ._common import MLModel


class RandomForest(MLModel):
    name, description = "RandomForest", "Bagged decision trees"

    def _fit(self, X, y, w, Xv, yv, wv, seed, k):
        from sklearn.ensemble import RandomForestClassifier
        self.m = RandomForestClassifier(S.FOREST_TREES, min_samples_leaf=3, max_features="sqrt", n_jobs=-1, random_state=seed).fit(X, y, sample_weight=w)
