import yaml
from modules.utils import get_filename
from modules.experiment import Experiment
import os

def main(args):

    if args['args']['model'] == 'logistic':
        print('logistic')

def read_args():
    args = None
    with open('args.yaml', "r") as file:
        args = yaml.full_load(file)
    
    args_path = os.path.join(os.getcwd(),"args.yaml")
    experiment = Experiment(args_path)

    main(args)

if __name__ == "__main__":
    read_args()

