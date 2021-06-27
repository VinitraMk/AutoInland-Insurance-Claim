from catboost import CatBoostClassifier

from modules.utils import get_model_params

class CatBoost:
    X = None
    y = None
    model = None

    def __init__(self, X, y, model):
        self.X = X
        self.y = y
        self.model = model

    def train_model(self, ensembler = False):
        params = {}
        if ensembler:
            params = get_model_params(ensembler, 'catboost')
        else:
            params = get_model_params()
        self.model = CatBoostClassifier(
                iterations = params['iterations'],
                learning_rate = params['learning_rate'],
                depth = params['depth'],
                l2_leaf_reg = params['l2_leaf_reg'],
                od_type = "Iter",
                verbose = 0
                )
        if not(ensembler):
            self.model.fit(self.X, self.y)
        return self.model

