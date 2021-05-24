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
        self.clean_age_col(int(self.preproc_args['missing']['age']))
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
        print('\nCleaning up Age Column')
        self.data['Age'] = self.data['Age'].apply(pd.to_numeric)
        self.X['Age'] = self.X['Age'].apply(pd.to_numeric)
        self.data['Age'] = self.data['Age'].mask(self.data['Age'] < 0, replace_value)
        self.X['Age'] = self.X['Age'].mask(self.X['Age'] < 0, replace_value)
        print('No of invalid ages: ',self.data[self.data['Age'] < 0]['Age'].shape)

    def plot_graph(self):
        for col in self.data.columns:
            if col in self.preproc_args['skip']:
                pass
            elif col == 'Age':
                col_names = ['Below 18', '18-45', '45-60', 'Above or equal to 60']
                data = {}
                data[col_names[0]] = { 
                    0: self.data[(self.data['Age'] < 18) & (self.data['target'] == 0)]['Age'].size,
                    1: self.data[(self.data['Age'] < 18) & (self.data['target'] == 1)]['Age'].size
                }
                data[col_names[1]] = {
                    0: self.data[((self.data['Age'] > 17) & (self.data['Age'] < 45)) & (self.data['target'] == 0)]['Age'].size,
                    1: self.data[((self.data['Age'] > 17) & (self.data['Age'] < 45)) & (self.data['target'] == 1)]['Age'].size
                }
                data[col_names[2]] = {
                    0: self.data[((self.data['Age'] > 44) & (self.data['Age'] < 60)) & (self.data['target'] == 0)]['Age'].size,
                    1: self.data[((self.data['Age'] > 44) & (self.data['Age'] < 60)) & (self.data['target'] == 1)]['Age'].size
                }
                data[col_names[3]] = {
                    0: self.data[(self.data['Age'] > 59) & (self.data['target'] == 0)]['Age'].size,
                    1: self.data[(self.data['Age'] > 59) & (self.data['target'] == 1)]['Age'].size
                }
                data = pd.DataFrame(data=list(data.values()), index=col_names)
                data.plot(kind="bar")
                plt.title("Age Distribution")
                plt.xlabel("Age Ranges")
                plt.ylabel("Count")
                plt.tight_layout()
                plt.savefig(f'{plot_path}/{col}_distribution.png')
                plt.clf()
            elif col == 'target':
                tdata = self.data
                unique_cols = tdata.groupby(col)['ID'].nunique()
                unique_cols.plot.bar(x=col,y="Count",title="Distribution of columns")
                plot_path = path.join(getenv("ROOT_DIR"),self.config['visualizations'])
                plt.tight_layout()
                plt.savefig(f'{plot_path}/{col}_distribution.png')
                plt.clf()
            else:
                tdata0 = self.data[self.data['target'] == 0]
                tdata1 = self.data[self.data['target'] == 1]
                unique_cols0 = tdata0.groupby(col)['ID'].nunique()
                unique_cols0 = unique_cols0.rename("0")
                unique_cols1 = tdata1.groupby(col)['ID'].nunique()
                unique_cols1 = unique_cols1.rename("1")
                unique_cols = pd.merge(unique_cols0, unique_cols1, right_index=True, left_index=True)
                unique_cols.plot(kind="bar")
                plot_path = path.join(getenv("ROOT_DIR"),self.config['visualizations'])
                plt.tight_layout()
                plt.savefig(f'{plot_path}/{col}_distribution.png')
                plt.clf()

        
