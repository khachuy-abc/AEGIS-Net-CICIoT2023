"""NaiveBayes - Gaussian Naive Bayes."""
from ._common import MLModel


class NaiveBayes(MLModel):
    name, description = "NaiveBayes", "Gaussian Naive Bayes"

    def _fit(self, X, y, w, Xv, yv, wv, seed, k):
        from sklearn.naive_bayes import GaussianNB
        self.m = GaussianNB().fit(X, y, sample_weight=w)
