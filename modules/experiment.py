import yaml
import os
import json
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from modules.utils import get_filename

class Experiment:
    yaml_path = ''
    model_args = {}
    config_args = {}
    preproc_args = {}
    test_features = None
    test_ids = None
    model = None
    validation_score = 0
    confusion_matrix = []

    def __init__(self, yaml_path, model):
        self.yaml_path = yaml_path
        with open(yaml_path) as file:
            args = yaml.full_load(file)
            self.config_args = args['config']
            self.model_args = args['args']
            self.preproc_args = args['preproc']
            self.model = model
        print(f'\nStarting experiment...')
        print('Model params:',self.model_args)

    def log_experiment(self,model_name, filename):
        model_path = os.path.join(self.config_args['log'],model_name)
        if not(os.path.exists(model_path)):
            os.mkdir(model_path)
        log_fnpath = os.path.join(model_path, f'{filename}_log.txt')
        log_file_contents = dict() 
        log_file_contents['model_args'] = self.model_args
        log_file_contents['model_output'] = { 'validation_score': self.validation_score,
                'confusion_matrix': self.confusion_matrix.tolist() }
        log_file_contents['preproc_args'] = self.preproc_args
        with open(log_fnpath, "w+") as file:
            file.write(json.dumps(log_file_contents, indent = 4))

    def validate(self, valid_X, valid_y):
        ypreds = self.model.predict(valid_X)
        self.validation_score = f1_score(valid_y, ypreds)
        self.confusion_matrix = confusion_matrix(valid_y, self.model.predict(valid_X))
        print('\nValidation set accuracy score: ', self.validation_score)
        print('\nConfusion matrix:\n', self.confusion_matrix,'\n')

    def predict_and_save_csv(self,test_features,test_ids):
        title = get_filename(self.model_args['model'])
        print(f'Saving predictions to {title}.csv...\n')
        y_preds = self.model.predict(test_features)
        y_ids = pd.DataFrame(test_ids, columns=['ID'])
        y_preds_df = pd.DataFrame(y_preds, columns=['target'])
        predictions = y_ids.join(y_preds_df)
        predictions.to_csv(f'{self.config_args["output"]}/{title}.csv', index = False)
        self.log_experiment(self.model_args['model'], title)

