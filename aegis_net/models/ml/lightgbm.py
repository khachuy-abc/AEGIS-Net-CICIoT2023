"""LightGBM - leaf-wise gradient-boosted trees; rounds are chosen by early stopping on the monitor split (no 'epochs')."""
import inspect

from ... import settings as S
from ._common import MLModel


class LightGBM(MLModel):
    name, description = "LightGBM", "Leaf-wise gradient-boosted trees (early stopping on the monitor split)"

    def _fit(self, X, y, w, Xv, yv, wv, seed, k):
        import lightgbm as lgb             # the library, not this file: imports are absolute
        self.m = lgb.LGBMClassifier(n_estimators=S.BOOST_MAX_ROUNDS, learning_rate=S.LGB_LEARNING_RATE, num_leaves=S.LGB_NUM_LEAVES,
                                    subsample=0.9, subsample_freq=1, colsample_bytree=0.9, n_jobs=-1, random_state=seed, verbose=-1)
        params = inspect.signature(self.m.fit).parameters      # LightGBM changed the name of the validation arguments between versions
        kw = dict(eval_X=Xv, eval_y=yv) if "eval_X" in params else dict(eval_set=[(Xv, yv)])
        wname = next((p for p in ("eval_sample_weight", "eval_weight") if p in params), None)
        if wname:
            kw[wname] = [wv]
        self.m.fit(X, y, sample_weight=w, callbacks=[lgb.early_stopping(S.BOOST_PATIENCE, verbose=False)], **kw)
        self.info = dict(best_round=int(self.m.best_iteration_ or S.BOOST_MAX_ROUNDS), round_cap=S.BOOST_MAX_ROUNDS)
