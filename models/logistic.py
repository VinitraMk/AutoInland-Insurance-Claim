from sklearn.model import LogisticRegressionCV

from modules.utils import get_model_params

class Logistic:
    X = None
    y = None
    model = None
    model_params = None

    def __init__(self,X,y,test_ids):
        self.X = X
        self.y = y
        self.model_params = get_model_params()

    def train_model(self):
        logistic = LogisticRegressionCV(scoring=self.model_params['scoring'],
                solver=self.model_params['solver'],
                tolerance=self.model_params['tolerance'],
                class_weight=self.model_params['class_weight'],
                n_jobs = -1,
                random_state = 42)
        self.model = logistic.fit(self.X, self.y)
        return self.model


