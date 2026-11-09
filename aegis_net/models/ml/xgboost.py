"""XGBoost - gradient-boosted trees; rounds are chosen by early stopping on the monitor split (no 'epochs')."""
from ... import settings as S
from ._common import MLModel


class XGBoost(MLModel):
    name, description = "XGBoost", "Gradient-boosted trees (early stopping on the monitor split)"

    def _fit(self, X, y, w, Xv, yv, wv, seed, k):
        import xgboost as xgb              # the library, not this file: imports are absolute
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        self.m = xgb.XGBClassifier(n_estimators=S.BOOST_MAX_ROUNDS, learning_rate=S.XGB_ETA, max_depth=S.XGB_MAX_DEPTH, subsample=0.9,
                                   colsample_bytree=0.9, tree_method="hist", device=device, early_stopping_rounds=S.BOOST_PATIENCE,
                                   random_state=seed, eval_metric="mlogloss" if k > 2 else "logloss", n_jobs=-1)
        self.m.fit(X, y, sample_weight=w, eval_set=[(Xv, yv)], sample_weight_eval_set=[wv], verbose=False)
        self.info = dict(best_round=int(self.m.best_iteration) + 1, round_cap=S.BOOST_MAX_ROUNDS)
