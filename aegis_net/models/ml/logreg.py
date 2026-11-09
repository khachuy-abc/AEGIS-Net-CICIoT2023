"""LogReg - multinomial logistic regression (the linear baseline)."""
from ._common import MLModel


class LogReg(MLModel):
    name, description = "LogReg", "Logistic regression (linear baseline)"

    def _fit(self, X, y, w, Xv, yv, wv, seed, k):
        from sklearn.linear_model import LogisticRegression
        self.m = LogisticRegression(C=1.0, max_iter=2000, tol=1e-3).fit(X, y, sample_weight=w)
