from modules.experiment import Experiment
from modules.utils import get_config, get_preproc_params

import pandas as pd
import numpy as np
from os import path, getenv
import matplotlib.pyplot as plt

class Preprocessor:
    X = None
    y = None
    data = None
    config = None
    preproc_args = None

    def __init__(self):
        self.config = get_config()
        data = pd.read_csv(path.join(self.config['input'],'Train.csv'))
        self.data = data
        self.X = data.drop(columns = ['target','ID'])
        self.y = data['target']
        self.preproc_args = get_preproc_params()

    def fill_missing_values(self):
        self.clean_gender_col(self.preproc_args['missing']['gender'])
        self.clean_age_col(self.preproc_args['missing']['age'])
        self.plot_graph()


    def replace_cols(self, col, to_replace, value):
        for val in to_replace:
            rp = self.data[col].replace(val,value)
            self.data[col] = rp
            self.X[col] = rp

    def clean_gender_col(self, replace_value):
        print('\nCleaning up Gender column')
        print('No of null:',self.data['Gender'].isnull().sum())
        self.replace_cols('Gender',['Entity','NO GENDER','NOT STATED','SEX','Joint Gender'], 'Other')
        print('No of null:',self.data['Gender'].isnull().sum())
        rpcol = self.data['Gender'].replace(to_replace=np.nan,
        value=replace_value)
        self.data['Gender'] = rpcol
        self.X['Gender'] = rpcol
        print('No of null:',self.data['Gender'].isnull().sum())

    def clean_age_col(self, replace_value):
        self.data = self.data.mask(self.data['Age'] < 0, replace_value)

    def plot_graph(self):
        for col in self.X.columns:
            if col in self.preproc_args['skip']:
                pass
            else:
                tdata = self.data
                unique_cols = tdata.groupby(col)['ID'].nunique()
                unique_cols.plot.bar(x=col,y="Count",title="Distribution of columns")
                plot_path = path.join(getenv("ROOT_DIR"),self.config['visualizations'])
                plt.tight_layout()
                plt.savefig(f'{plot_path}/{col}_distribution.png')

        
