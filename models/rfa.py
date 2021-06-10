from sklearn.ensemble import RandomForestClassifier
from modules.utils import get_model_params
class RandomForest:
    model = None
    X = None
    y = None

    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y

    def train_model(self):
        params = get_model_params()
        self.model = RandomForestClassifier(
                n_estimators = params['n_estimators'],
                criterion = params['criterion'],
                max_features = params['max_features'],
                n_jobs = -1,
                random_state = 42,
                class_weight = params['class_weight'])
        self.model.fit(self.X, self.y)
        return self.model

