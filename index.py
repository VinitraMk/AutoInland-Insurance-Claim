import yaml
import os

from modules.utils import get_config_path, get_model_params, get_preproc_params, get_validation_params
from modules.experiment import Experiment
from modules.preprocessing import Preprocessor
from modules.validation import Validate
from models.logistic import Logistic
from models.knn import KNN
from models.svm import SVM

def main(args, val_args):

    preprocessor = Preprocessor()
    data, test_features, test_ids = preprocessor.start_preprocessing()
    model = None
    avg_score = 0

    for i in range(val_args['k']):
        validate = Validate(data)
        X, y, valid_X, valid_y = validate.prepare_dataset()
        if args['model'] == 'logistic':
            logistic = Logistic(X,y, model)
            model = logistic.train_model()
        elif args['model'] == 'knn':
            knn = KNN(X,y, model)
            model = knn.train_model()
        elif args['model'] == 'svm':
            svm = SVM(X,y, model)
            model = svm.train_model()
        experiment = Experiment(get_config_path(), model)
        score = experiment.validate(valid_X, valid_y)
        avg_score = score + avg_score
    avg_score = avg_score / val_args['k']
    print('\nAverage F1 score of the model:',avg_score,'\n')
    experiment.predict_and_save_csv(test_features, test_ids, avg_score)


def read_args():
    args = get_model_params()
    val_args = get_validation_params()
    main(args, val_args)

def set_root_dir():
    if not(os.getenv('ROOT_DIR')):
        os.environ['ROOT_DIR'] = os.getcwd()

if __name__ == "__main__":
    set_root_dir()
    read_args()

