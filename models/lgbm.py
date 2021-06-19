from modules.utils import get_model_params

from lightgbm import LGBMClassifier

class LightGBM:
    X = None
    y = None
    model = None

    def __init__(self,X, y, model):
        self.X = X
        self.y = y
        self.model = model

    def train_model(self):
        params = get_model_params()
        self.model = LGBMClassifer(random_state = 42,
                class_weight=params['class_weight'],
                objective = 'multiclass',
                num_class = 2,
                n_estimators = params['n_estimators'],
                learning_rate = params['learning_rate'])
        self.model.fit(self.X, self.y)
        return self.model
