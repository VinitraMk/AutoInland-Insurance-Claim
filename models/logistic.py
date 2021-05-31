from sklearn.linear_model import LogisticRegression

from modules.utils import get_model_params

class Logistic:
    X = None
    y = None
    model = None
    model_params = None

    def __init__(self,X,y):
        self.X = X
        self.y = y
        self.model_params = get_model_params()

    def train_model(self):
        print('\nTraining a logistic classifier')
        logistic = LogisticRegression(solver=self.model_params['solver'],
                class_weight=self.model_params['class_weight'],
                n_jobs = -1,
                random_state = 42,
                tol=self.model_params['tolerance'],
                C=self.model_params['C'])
        self.model = logistic.fit(self.X, self.y)
        return self.model


