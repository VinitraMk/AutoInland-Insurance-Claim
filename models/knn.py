from sklearn.neighbors import KNeighborsClassifier
from modules.utils import get_model_params

class KNN:
    X = None
    y = None
    model_params = None

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model_params = get_model_params()

    def train_model(self):
        print('\nTraining KNN Classifier...')
        knn = KNeighborsClassifier(n_jobs = -1,
                n_neighbors = self.model_params['k'],
                weights = self.model_params['weights'],
                algorithm = self.model_params['algorithm'],
                leaf_size = self.model_params['leaf_size'],
                p = self.model_params['p'],
                metric = self.model_params['metric'])
        knn.fit(self.X, self.y)
        return knn
