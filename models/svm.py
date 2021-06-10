from sklearn.svm import SVC, LinearSVC

from modules.utils import get_model_params
from modules.validation import Validate

class SVM:
    X = None
    y = None
    model = None

    def __init__(self, X, y, model):
        self.model = model
        self.X = X
        self.y = y

    def train_model(self):
        params = get_model_params()
        if params['kernel'] != 'linear':
            self.model = SVC(C=params['C'],
                    kernel=params['kernel'],
                    degree=params['degree'],
                    gamma=params['gamma'],
                    tol=params['tol'],
                    class_weight=params['class_weight'],
                    max_iter=-1,
                    random_state = 42)
        else:
            self.model = LinearSVC(C=params['C'],
                    loss=params['loss'],
                    penalty=params['penalty'],
                    class_weight=params['class_weight'],
                    random_state = 42,
                    max_iter = params['max_iter'],
                    dual = params['dual']
                    )

        self.model = self.model.fit(self.X,self.y)
        return self.model


