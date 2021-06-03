from sklearn.svm import SVC
from modules.utils import get_model_params

class SVM:
    X = None
    y = None
    model = None

    def __init__(self, X, y):
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
        self.model = self.model.fit(self.X,self.y)
        return self.model


