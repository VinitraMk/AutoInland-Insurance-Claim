import yaml
import os

from modules.utils import get_config_path, get_model_params, get_preproc_params
from modules.experiment import Experiment
from modules.preprocessing import Preprocessor

def main(args):

    preprocessor = Preprocessor()
    preprocessor.fill_missing_values()

def read_args():
    experiment = Experiment(get_config_path())
    args = get_model_params()
    main(args)

def set_root_dir():
    if not(os.getenv('ROOT_DIR')):
        os.environ['ROOT_DIR'] = os.getcwd()

if __name__ == "__main__":
    set_root_dir()
    read_args()

