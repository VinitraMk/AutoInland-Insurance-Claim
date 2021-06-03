import yaml
import os

from modules.utils import get_config_path, get_model_params, get_preproc_params
from modules.experiment import Experiment
from modules.preprocessing import Preprocessor
from models.logistic import Logistic
from models.knn import KNN
from models.svm import SVM

def main(args):

    preprocessor = Preprocessor()
    X, y, valid_X, valid_y, test_features, test_ids = preprocessor.start_preprocessing()
    model = None
    
    if args['model'] == 'logistic':
        logistic = Logistic(X,y)
        model = logistic.train_model()
    elif args['model'] == 'knn':
        knn = KNN(X,y)
        model = knn.train_model()
    elif args['model'] == 'svm':
        svm = SVM(X,y)
        model = svm.train_model()

    experiment = Experiment(get_config_path(), model)
    experiment.validate(valid_X, valid_y)
    experiment.predict_and_save_csv(test_features, test_ids)


def read_args():
    args = get_model_params()
    main(args)

def set_root_dir():
    if not(os.getenv('ROOT_DIR')):
        os.environ['ROOT_DIR'] = os.getcwd()

if __name__ == "__main__":
    set_root_dir()
    read_args()

