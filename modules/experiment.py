import yaml
import os
import json

class Experiment:
    yaml_path = ''
    model_args = {}
    config_args = {}

    def __init__(self, yaml_path):
        self.yaml_path = yaml_path
        with open(yaml_path) as file:
            args = yaml.full_load(file)
            self.config_args = args['config']
            self.model_args = args['args']
            print('Model params:',args)

    def log_experiment(self,model_name, filename):
        model_path = os.path.join(self.config_args['config']['log'],model_name)
        if not(os.path.exists(model_path)):
            os.mkdir(model_path)
        log_fnpath = os.path.join(model_path, f'{filename}_log.txt')
        with open(log_fnpath, "w+") as file:
            file.write(json.dumps(model_args))
    
    def get_model_params(cls):
        return cls.model_args

    def get_config(cls):
        return cls.config_args


