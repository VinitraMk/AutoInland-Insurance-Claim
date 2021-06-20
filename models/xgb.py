from xgboost import XGBClassifier

import pandas as pd
from modules.utils import get_model_params

class XGB:
    X = None
    y = None
    model = None

    def __init__(self, X, y, model):
        self.X = X
        self.y = y
        self.model = model

    def train_model(self):
        params = get_model_params()
        self.model = XGBClassifier(random_state = 42,
                objective = 'multi:softmax',
                eval_metric = 'merror',
                use_label_encoder = False,
                learning_rate = params['learning_rate'],
                n_estimators = params['n_estimators'],
                num_class=2,
                min_split_loss = params['min_split_loss'],
                tree_method = params['tree_method'],
                grow_policy = params['grow_policy'],
                single_precision_histogram = params['single_precision_histogram'],
                nthread=params['nthread'])
        self.model = self.model.fit(self.X, self.y)
        return self.model
