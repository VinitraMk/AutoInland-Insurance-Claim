import yaml
import os
import json

class Experiment:
    input_path = ''
    output_path = ''
    log_path = ''
    yaml_path = ''
    model_args = {}

    def __init__(self, yaml_path):
        self.yaml_path = yaml_path
        with open(yaml_path) as file:
            args = yaml.full_load(file)
            self.input_path, self.output_path, self.log_path = [args['config'][k] for k in args['config'].keys()]
            self.model_args = args['args']
            print('Model params:',args)

    def log_experiment(self,model_name, filename):
        model_path = os.path.join(log_path,model_name)
        if not(os.path.exists(model_path)):
            os.mkdir(model_path)
        log_fnpath = os.path.join(model_path, f'{filename}_log.txt')
        with open(log_fnpath, "w+") as file:
            file.write(json.dumps(model_args))
        
    def get_model_params(self):
        return self.model_args


