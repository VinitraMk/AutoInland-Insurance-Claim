from modules.utils import get_model_params

from lightgbm import LGBMClassifier
import re

class LightGBM:
    X = None
    y = None
    model = None

    def __init__(self,X, y, model):
        self.X = X
        self.y = y
        self.model = model

    def train_model(self, ensembler = False):
        params = {}
        if ensembler:
            params = get_model_params(ensembler, 'lgbm')
        else:
            params = get_model_params()

        #self.X = self.X.rename(columns = lambda x: re.sub('[^A-Za-z0-9_]+','',x))
        self.model = LGBMClassifier(random_state = 42,
                class_weight=params['class_weight'],
                n_estimators = params['n_estimators'],
                learning_rate = params['learning_rate'],
                min_split_gain = params['min_split_loss'],
                num_leaves = params['num_leaves'],
                min_child_samples = params['min_child_samples'],
                min_child_weight = params['min_child_weight'])
        if not(ensembler):
            self.model.fit(self.X, self.y)
        return self.model
