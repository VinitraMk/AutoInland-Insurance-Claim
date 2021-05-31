from modules.experiment import Experiment
from modules.utils import get_config, get_preproc_params

import pandas as pd
import numpy as np
from os import path, getenv
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import LabelEncoder

class Preprocessor:
    X = None
    y = None
    data = None
    test = None
    valid_X = None
    valid_y = None
    config = None
    preproc_args = None

    def __init__(self):
        self.config = get_config()
        data = pd.read_csv(path.join(self.config['input'],'Train.csv'))
        self.data = data
        self.test = pd.read_csv(path.join(self.config['input'], 'Test.csv'))
        self.test_ids = self.test['ID']
        self.preproc_args = get_preproc_params()

    def drop_skip_columns(self):
        self.data = self.data.drop(columns=self.preproc_args['skip'])
        self.test = self.test.drop(columns=self.preproc_args['skip'])

    def start_preprocessing(self):
        print('\nCleaning up columns...')
        self.clean_gender_col(self.preproc_args['missing']['gender'])
        self.clean_age_col(int(self.preproc_args['missing']['age']))
        self.clean_color_col(self.preproc_args['missing']['color'])
        self.clean_carmake_col()
        self.clean_carcat_col()
        print('\nPlotting distribution...')
        self.plot_graph()
        self.drop_skip_columns()
        print('\nSplitting datasets to train, validation and test sets')
        self.do_data_split()
        print('\nApplying label encoder...')
        return self.encode_labels()

    def do_data_split(self):
        valid_len = math.ceil(0.2 * self.data.shape[0])
        train_len = self.data.shape[0] - valid_len
        self.X = self.data[0:train_len]
        self.y = self.X['target']
        self.X = self.X.drop(columns=['target'])
        self.valid_X = self.data[train_len:]
        self.valid_y = self.valid_X['target']
        self.valid_X = self.valid_X.drop(columns=['target'])
        print('Lengths of test, train, valid',self.test.shape, self.X.shape, self.valid_X.shape)

    def encode_labels(self):
        for col in self.X.columns:
            if col != 'Age' and col != 'No_Pol':
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col])
                self.X[col] = le.transform(self.X[col])
                self.valid_X[col] = le.transform(self.valid_X[col])
                self.test[col] = self.test[col].map(lambda s: 'unknown' if s not in le.classes_ else s)
                if self.test[self.test[col] == 'unknown'].shape[0] != 0:
                    le.classes_ = np.append(le.classes_,'unknown')
                self.test[col] = le.transform(self.test[col])
                print(f'Encoded {col}', le.classes_)

        return self.X, self.y, self.valid_X, self.valid_y, self.test, self.test_ids


    def replace_cols(self, col, to_replace, value):
        for val in to_replace:
            rp = self.data[col].replace(val,value)
            self.data[col] = rp
            self.test[col] = rp

    def clean_gender_col(self, replace_value):
        self.replace_cols('Gender',['Entity','NO GENDER','NOT STATED','SEX','Joint Gender'], 'Other')
        rpcol = self.data['Gender'].replace(to_replace=np.nan,
        value=replace_value)
        self.data['Gender'] = rpcol
        self.test['Gender'] = rpcol

    def clean_age_col(self, replace_value):
        self.data['Age'] = self.data['Age'].apply(pd.to_numeric)
        self.test['Age'] = self.test['Age'].apply(pd.to_numeric)
        self.data['Age'] = self.data['Age'].mask(self.data['Age'] < 0, replace_value)
        self.test['Age'] = self.test['Age'].mask(self.test['Age'] < 0, replace_value)
        self.data['Age'] = self.data['Age'].mask(self.data['Age'] > 120, "120")
        self.test['Age'] = self.test['Age'].mask(self.test['Age'] > 120, "120")
        self.data['Age'] = self.data['Age'].astype(str)
        self.test['Age'] = self.test['Age'].astype(str)


    def clean_color_col(self, color_mappings):
        self.data['Subject_Car_Colour'] = self.data['Subject_Car_Colour'].str.replace(" ","")
        self.test['Subject_Car_Colour'] = self.test['Subject_Car_Colour'].str.replace(" ","")
        self.data['Subject_Car_Colour'] = self.data['Subject_Car_Colour'].replace('AsAttached',"0")
        self.test['Subject_Car_Colour'] = self.test['Subject_Car_Colour'].replace('AsAttached',"0")
        self.data['Subject_Car_Colour'] = self.data['Subject_Car_Colour'].fillna("0")
        self.test['Subject_Car_Colour'] = self.test['Subject_Car_Colour'].fillna("0")


        for color in color_mappings:
            self.data['Subject_Car_Colour'] = self.data['Subject_Car_Colour'].replace(color, color_mappings[color])
            self.test['Subject_Car_Colour'] = self.test['Subject_Car_Colour'].replace(color, color_mappings[color])
             
    def clean_carmake_col(self):
        self.data['Subject_Car_Make'] = self.data['Subject_Car_Make'].replace('.',np.nan)
        self.test['Subject_Car_Make'] = self.test['Subject_Car_Make'].replace('.',np.nan)
        self.data['Subject_Car_Make'] = self.data['Subject_Car_Make'].fillna("0")
        self.test['Subject_Car_Make'] = self.test['Subject_Car_Make'].fillna("0")

    def clean_carcat_col(self):
        self.data['Car_Category'] = self.data['Car_Category'].fillna("0")
        self.test['Car_Category'] = self.test['Car_Category'].fillna("0")

    def plot_graph(self):
        for col in self.data.columns:
            if col in self.preproc_args['skip']:
                pass
            elif col == 'Age':
                col_names = ['Below 18', '18-45', '45-60', 'Above or equal to 60']
                data = {}
                data[col_names[0]] = { 
                    0: self.data[(self.data['Age'] < "18") & (self.data['target'] == 0)]['Age'].size,
                    1: self.data[(self.data['Age'] < "18") & (self.data['target'] == 1)]['Age'].size
                }
                data[col_names[1]] = {
                    0: self.data[((self.data['Age'] > "17") & (self.data['Age'] < "45")) & (self.data['target'] == 0)]['Age'].size,
                    1: self.data[((self.data['Age'] > "17") & (self.data['Age'] < "45")) & (self.data['target'] == 1)]['Age'].size
                }
                data[col_names[2]] = {
                    0: self.data[((self.data['Age'] > "44") & (self.data['Age'] < "60")) & (self.data['target'] == 0)]['Age'].size,
                    1: self.data[((self.data['Age'] > "44") & (self.data['Age'] < "60")) & (self.data['target'] == 1)]['Age'].size
                }
                data[col_names[3]] = {
                    0: self.data[(self.data['Age'] > "59") & (self.data['target'] == 0)]['Age'].size,
                    1: self.data[(self.data['Age'] > "59") & (self.data['target'] == 1)]['Age'].size
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

