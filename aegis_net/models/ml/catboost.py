"""CatBoost - ordered gradient boosting on symmetric trees; rounds are chosen by early stopping on the monitor split."""
from ... import settings as S
from ._common import MLModel


class CatBoost(MLModel):
    name, description = "CatBoost", "Ordered boosting on symmetric trees (early stopping on the monitor split)"

    def _fit(self, X, y, w, Xv, yv, wv, seed, k):
        from catboost import CatBoostClassifier, Pool   # the library, not this file: imports are absolute
        self.m = CatBoostClassifier(iterations=S.CATBOOST_ROUNDS, learning_rate=S.CATBOOST_LR, depth=S.CATBOOST_DEPTH,
                                    loss_function="MultiClass" if k > 2 else "Logloss", random_seed=seed, thread_count=-1, verbose=0,
                                    early_stopping_rounds=S.BOOST_PATIENCE, allow_writing_files=False)
        self.m.fit(Pool(X, y, weight=w), eval_set=Pool(Xv, yv, weight=wv))
        self.info = dict(best_round=int(self.m.get_best_iteration()) + 1, round_cap=S.CATBOOST_ROUNDS)
