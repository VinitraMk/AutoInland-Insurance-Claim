from modules.experiment import Experiment
from modules.utils import get_config, get_preproc_params
from modules.validation import Validate

import pandas as pd
import numpy as np
from os import path, getenv
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import RandomOverSampler, SMOTE

class Preprocessor:
    X = None
    y = None
    data = None
    test = None
    valid_X = None
    valid_y = None
    config = None
    preproc_args = None
    missing_val_maps = dict()

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
        self.find_missing_val_replacements()
        self.clean_gender_col(self.preproc_args['missing']['gender'])
        self.clean_age_col(int(self.preproc_args['missing']['age']), self.preproc_args['missing']['age_map'])
        self.clean_color_col(self.preproc_args['missing']['color'], self.preproc_args['missing']['color_rp'])
        self.clean_carmake_col(self.preproc_args['missing']['make'])
        self.clean_carcat_col(self.preproc_args['missing']['category'])
        self.clean_state_col(self.preproc_args['missing']['state'])
        self.clean_LGA_col(self.preproc_args['missing']['lga'])
        print('\nPlotting distribution...')
        self.plot_graph()
        self.drop_skip_columns()
        print('\nApplying label encoder...')
        self.encode_labels()
        return self.apply_oversampling()

    def correcting_states(self):
        config = get_config_params()
        state_data = pd.read_csv(config['state_mapping'])
        unique_vals = self.data['State'].unique()
        for val in unique_vals:
            lga = state_data[state_data['State'] == val].iloc[0]['LGA']

    def find_missing_val_replacements(self):
        for col in self.data.columns:
            if col != 'ID':
                mode = self.data[col].mode()
                self.missing_val_maps[col] = mode
        print(self.missing_val_maps)

    def apply_oversampling(self):
        over_sampler = SMOTE(sampling_strategy=self.preproc_args['sampling_strategy'], random_state = 42,
                k_neighbors = self.preproc_args['sampling_k'],
                n_jobs=-1)
        X = self.data
        y = self.data['target']
        X = X.drop(columns=['target'])
        X, y = over_sampler.fit_resample(X, y)
        self.data = X.join(y)
        print('\nData shape after sampling', self.data.shape,'\n')
        return self.data, self.test, self.test_ids

    def do_data_split(self):
        validate = Validate(self.data)
        self.X, self.y, sel.valid_X, self.valid_y = validate.prepare_dataset()
        print('Lengths of test, train, valid',self.test.shape, self.X.shape, self.valid_X.shape)

    def encode_labels(self):
        selected_cols = [x for x in self.data.columns if x != 'target' and x != 'Age' and x != 'No_Pol' and x not in self.preproc_args['skip']]
        le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int64)
        self.data[selected_cols] = le.fit_transform(self.data[selected_cols])
        self.test[selected_cols] = le.transform(self.test[selected_cols])
        print(f'Encoded classes', le.categories_)

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

    def clean_age_col(self, replace_value, age_map):
        self.data['Age'] = self.data['Age'].apply(pd.to_numeric)
        self.test['Age'] = self.test['Age'].apply(pd.to_numeric)
        self.data['Age'] = self.data['Age'].mask(self.data['Age'] < 0, replace_value)
        self.test['Age'] = self.test['Age'].mask(self.test['Age'] < 0, replace_value)
        '''
        self.data['Age_Cat'] = self.data['Age'].astype(str)
        self.test['Age_Cat'] = self.test['Age'].astype(str)
        self.data.loc[(self.data['Age'] < 19), 'Age_Cat'] = age_map[0]
        self.test.loc[(self.test['Age'] < 19), 'Age_Cat'] = age_map[0]
        self.data.loc[(self.data['Age'] > 18) & (self.data['Age'] < 35), 'Age_Cat'] = age_map[19]
        self.test.loc[(self.test['Age'] > 18) & (self.test['Age'] < 35), 'Age_Cat'] = age_map[19]
        self.data.loc[(self.data['Age'] > 34) & (self.data['Age'] < 60), 'Age_Cat'] = age_map[35]
        self.test.loc[(self.test['Age'] > 34) & (self.test['Age'] < 60), 'Age_Cat'] = age_map[35]
        self.data.loc[(self.data['Age'] > 59), 'Age_Cat'] = age_map[60]
        self.test.loc[(self.test['Age'] > 59), 'Age_Cat'] = age_map[60]
        '''

    def clean_color_col(self, color_mappings, color_rp):
        self.data['Subject_Car_Colour'] = self.data['Subject_Car_Colour'].str.replace(" ","")
        self.test['Subject_Car_Colour'] = self.test['Subject_Car_Colour'].str.replace(" ","")
        self.data['Subject_Car_Colour'] = self.data['Subject_Car_Colour'].replace('AsAttached',"0")
        self.test['Subject_Car_Colour'] = self.test['Subject_Car_Colour'].replace('AsAttached',"0")
        self.data['Subject_Car_Colour'] = self.data['Subject_Car_Colour'].fillna("0")
        self.test['Subject_Car_Colour'] = self.test['Subject_Car_Colour'].fillna("0")
        self.data['Subject_Car_Colour'] = self.data['Subject_Car_Colour'].replace('0',color_rp)
        self.test['Subject_Car_Colour'] = self.test['Subject_Car_Colour'].replace('0',color_rp)
        for color in color_mappings:
            self.data['Subject_Car_Colour'] = self.data['Subject_Car_Colour'].replace(color, color_mappings[color])
            self.test['Subject_Car_Colour'] = self.test['Subject_Car_Colour'].replace(color, color_mappings[color])
             
    def clean_carmake_col(self, replace_value):
        self.data['Subject_Car_Make'] = self.data['Subject_Car_Make'].replace('.',np.nan)
        self.test['Subject_Car_Make'] = self.test['Subject_Car_Make'].replace('.',np.nan)
        self.data['Subject_Car_Make'] = self.data['Subject_Car_Make'].fillna(replace_value)
        self.test['Subject_Car_Make'] = self.test['Subject_Car_Make'].fillna(replace_value)

    def clean_carcat_col(self, replace_value):
        self.data['Car_Category'] = self.data['Car_Category'].fillna(replace_value)
        self.test['Car_Category'] = self.test['Car_Category'].fillna(replace_value)

    def clean_state_col(self, replace_value):
        self.data['State'] = self.data['State'].fillna(replace_value)
        self.test['State'] = self.test['State'].fillna(replace_value)

    def clean_LGA_col(self, replace_value):
        self.data['LGA_Name'] = self.data['LGA_Name'].fillna(replace_value)
        self.test['LGA_Name'] = self.test['LGA_Name'].fillna(replace_value)

    def plot_graph(self):
        for col in self.data.columns:
            if col in self.preproc_args['skip']:
                pass
            elif col == 'Age':
                col_names = ['Below 18', '18-34', '35-59', 'Above or equal to 60']
                data = {}
                data[col_names[0]] = { 
                    0: self.data[(self.data['Age'] < 18) & (self.data['target'] == 0)]['Age'].size,
                    1: self.data[(self.data['Age'] < 18) & (self.data['target'] == 1)]['Age'].size
                }
                data[col_names[1]] = {
                    0: self.data[((self.data['Age'] > 17) & (self.data['Age'] < 34)) & (self.data['target'] == 0)]['Age'].size,
                    1: self.data[((self.data['Age'] > 17) & (self.data['Age'] < 34)) & (self.data['target'] == 1)]['Age'].size
                }
                data[col_names[2]] = {
                    0: self.data[((self.data['Age'] > 34) & (self.data['Age'] < 60)) & (self.data['target'] == 0)]['Age'].size,
                    1: self.data[((self.data['Age'] > 34) & (self.data['Age'] < 60)) & (self.data['target'] == 1)]['Age'].size
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
                plt.figure(figsize=(10,6), dpi=80)
                tdata = self.data
                full_len = self.data.shape[0]
                unique_cols = tdata.groupby(col)['ID'].nunique()
                print(unique_cols)
                unique_cols.plot.bar(x=col,y="Count",title="Distribution of columns")
                for i, val in enumerate(list(unique_cols)):
                    plt.text(i, val/full_len, str(val/full_len))
                plot_path = path.join(getenv("ROOT_DIR"),self.config['visualizations'])
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

