from modules.experiment import Experiment
from modules.utils import get_config, get_preproc_params, break_date, extract_date, is_null, save_fig, get_model_params
from modules.validation import Validate

import pandas as pd
import numpy as np
from os import path, getenv
import matplotlib.pyplot as plt
import math
import re
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, PolynomialFeatures
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import NearMiss, TomekLinks, OneSidedSelection
from datetime import datetime
from sklearn.feature_selection import SelectKBest, chi2
import seaborn as sns

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
    state_map = None
    selected_cols = []

    def __init__(self):
        self.config = get_config()
        data = pd.read_csv(path.join(self.config['input'],'Train.csv'))
        self.data = data
        self.test = pd.read_csv(path.join(self.config['input'], 'Test.csv'))
        self.state_map = pd.read_csv(path.join(self.config['input'], 'NigerianStateNames.csv'))
        self.test_ids = self.test['ID']
        self.preproc_args = get_preproc_params()

    def drop_skip_columns(self):
        if not(self.preproc_args['select_kbest']):
            self.data = self.data.drop(columns=self.preproc_args['skip'])
            self.test = self.test.drop(columns=self.preproc_args['skip'])
        else:
            columns = list(set(self.data.columns) - set(self.selected_cols))
            if 'target' in columns:
                columns.remove('target')
            self.data = self.data.drop(columns=columns)
            self.test = self.test.drop(columns=columns)

    def start_preprocessing(self):
        print('\nCleaning up columns...')
        model_params = get_model_params()
        self.remake_date_cols()
        self.remake_states_lga()
        self.find_missing_val_replacements()
        self.clean_gender_col(self.preproc_args['missing']['gender'])
        self.clean_age_col(int(self.preproc_args['missing']['age']), self.preproc_args['missing']['age_map'])
        self.clean_color_col(self.preproc_args['missing']['color'])
        self.clean_carmake_col(self.preproc_args['missing']['make'])
        self.clean_carcat_col(self.preproc_args['missing']['category'])
        #self.clean_car_product(self.preproc_args['missing']['product'])
        #self.clean_state_col(self.preproc_args['missing']['state'])
        #self.clean_LGA_col(self.preproc_args['missing']['lga'])
        self.clean_date_col()
        #self.remake_nopol_col()
        print('\nPlotting distribution...')
        self.plot_graph()
        print('\nApplying label encoder...')
        #self.find_feature_correlation(self.preproc_args['correlation_LB'],self.preproc_args['correlation_UB'])
        self.select_best_features(model_params['feature_k'])
        self.drop_skip_columns()
        self.encode_labels()
        return self.apply_oversampling()


    def find_missing_val_replacements(self):
        for col in self.data.columns:
            if col != 'ID':
                mode = self.data[col].mode()
                self.missing_val_maps[col] = mode
        print('Missing value modes', self.missing_val_maps)

    def apply_oversampling(self):
        over_sampler = SMOTE(sampling_strategy=self.preproc_args['over_sampling_strategy'], random_state = 42,
                k_neighbors = self.preproc_args['over_sampling_k'],
                n_jobs=-1)
        #under_sampler = OneSidedSelection(sampling_strategy = 'majority', random_state=42, n_neighbors = 5, n_jobs = -1)
        X = self.data.drop(columns = ['target'])
        y = self.data['target']
        X, y = over_sampler.fit_resample(X, y)
        self.data = X.join(y)
        print('\nData shape after sampling', self.data.shape)
        print('Test Data shape', self.test.shape,'\n')
        self.data = self.data.rename(columns = lambda x: re.sub('[^A-Za-z0-9_]+','',x))
        self.test = self.test.rename(columns = lambda x: re.sub('[^A-Za-z0-9_]+','',x))
        return self.data, self.test, self.test_ids

    def select_best_features(self, feature_k):
        feature_names = [x for x in self.data.columns if x not in self.preproc_args['skip']]
        if 'target' in feature_names:
            feature_names.remove('target')
        data = self.apply_label_encoding(self.data)
        kbest = SelectKBest(chi2, k = feature_k)
        X = data.drop(columns=['target'] + self.preproc_args['skip'])
        y = data['target']
        X = kbest.fit_transform(X, y)
        test_X = self.test.drop(columns=self.preproc_args['skip'])
        test_X = kbest.transform(test_X)
        mask = kbest.get_support()

        for is_feature_selected, feature in zip(mask, feature_names):
            if is_feature_selected:
                self.selected_cols.append(feature)
        print('\nSelected features: ',list(self.selected_cols))

    def find_feature_correlation(self, LB, UB):
        data = self.apply_label_encoding(self.data)
        plt.figure(figsize=(20,20))
        corr = data.corr()
        print(corr['target'].sort_values(ascending = False))
        sns.heatmap(corr, annot=True,cmap=plt.cm.Reds)
        save_fig('correlation_heatmap',plt)
        relevant_features = pd.concat([corr[corr['target'] > LB],corr[corr['target'] < UB]])
        print('\nRelevant Features')
        print(relevant_features)

        for row in relevant_features.iterrows():
            correlation_list = row[1]
            print(f'Correlations with: {row[0]}\n',row[1].sort_values(),'\n')
        #exit()

    def encode_labels(self):
        selected_cols = []
        date_cols = ['Policy_Start_Day', 'Policy_Start_Month', 'Policy_Start_Year', 'Policy_End_Day', 'Policy_End_Month', 'Policy_End_Year', 'Transaction_Day', 'Transaction_Month', 'Transaction_Year']
        other_cols = ['No_Pol', 'Age', 'target', 'ID']
        if not(self.preproc_args['select_kbest']):
            selected_cols = [x for x in self.data.columns if x not in other_cols and x not in self.preproc_args['skip'] and x not in date_cols]
        else:
            selected_cols = [x for x in self.selected_cols if x not in date_cols and x not in other_cols]
        train_len = self.data.shape[0]
        X = self.data.drop(columns = ['target'])
        y = self.data['target']
        all_data = pd.concat([X, self.test]).reset_index(drop = True)
        all_data = pd.get_dummies(data = all_data, columns = selected_cols)
        self.data = all_data[:train_len]
        self.data = self.data.join(y)
        self.test = all_data[train_len:]
        print(f'\nSelected columns: {selected_cols}\n')

    def apply_label_encoding(self, data):
        date_cols = ['Policy_Start_Day', 'Policy_Start_Month', 'Policy_Start_Year', 'Policy_End_Day', 'Policy_End_Month', 'Policy_End_Year', 'Transaction_Day', 'Transaction_Month', 'Transaction_Year']
        other_cols = ['No_Pol', 'Age', 'target', 'ID']
        selected_cols = [x for x in data.columns if x not in other_cols and x not in self.preproc_args['skip'] and x not in date_cols]

        for col in selected_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])

        return data

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
        self.data['Age'] = self.data['Age'].mask(self.data['Age'] > 105, replace_value)
        self.test['Age'] = self.test['Age'].mask(self.test['Age'] > 105, replace_value)
        
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

    def clean_color_col(self, color_rp):
        self.data['Subject_Car_Colour'] = self.data['Subject_Car_Colour'].str.replace(" ","")
        self.test['Subject_Car_Colour'] = self.test['Subject_Car_Colour'].str.replace(" ","")
        self.data['Subject_Car_Colour'] = self.data['Subject_Car_Colour'].str.replace(".","")
        self.test['Subject_Car_Colour'] = self.test['Subject_Car_Colour'].str.replace(".","")
        self.data['Subject_Car_Colour'] = self.data['Subject_Car_Colour'].str.replace("&","")
        self.test['Subject_Car_Colour'] = self.test['Subject_Car_Colour'].str.replace("&","")
        self.data['Subject_Car_Colour'] = self.data['Subject_Car_Colour'].replace('AsAttached',"0")
        self.test['Subject_Car_Colour'] = self.test['Subject_Car_Colour'].replace('AsAttached',"0")
        self.data['Subject_Car_Colour'] = self.data['Subject_Car_Colour'].fillna("0")
        self.test['Subject_Car_Colour'] = self.test['Subject_Car_Colour'].fillna("0")
        self.data['Subject_Car_Colour'] = self.data['Subject_Car_Colour'].replace('0',color_rp)
        self.test['Subject_Car_Colour'] = self.test['Subject_Car_Colour'].replace('0',color_rp)
             
    def clean_carmake_col(self, replace_value):
        self.data['Subject_Car_Make'] = self.data['Subject_Car_Make'].replace('.',np.nan)
        self.test['Subject_Car_Make'] = self.test['Subject_Car_Make'].replace('.',np.nan)
        self.data['Subject_Car_Make'] = self.data['Subject_Car_Make'].replace('As Attached',np.nan)
        self.test['Subject_Car_Make'] = self.test['Subject_Car_Make'].replace('As Attached',np.nan)
        self.data['Subject_Car_Make'] = self.data['Subject_Car_Make'].replace('Land Rover.','Land Rover')
        self.test['Subject_Car_Make'] = self.test['Subject_Car_Make'].replace('Land Rover.','Land Rover')
        self.data['Subject_Car_Make'] = self.data['Subject_Car_Make'].fillna(replace_value)
        self.test['Subject_Car_Make'] = self.test['Subject_Car_Make'].fillna(replace_value)
        self.data['Subject_Car_Make'] = self.data['Subject_Car_Make'].apply(lambda x: x.title().replace(" ",""))
        self.test['Subject_Car_Make'] = self.test['Subject_Car_Make'].apply(lambda x: x.title().replace(" ",""))

    def clean_carcat_col(self, replace_value):
        self.data['Car_Category'] = self.data['Car_Category'].fillna(replace_value)
        self.test['Car_Category'] = self.test['Car_Category'].fillna(replace_value)
        self.data['Car_Category'] = self.data['Car_Category'].replace('Pick Up > 3 Tons','Pick Up Gt Than Three Tons')
        self.test['Car_Category'] = self.test['Car_Category'].replace('Pick Up > 3 Tons','Pick Up Gt Than Three Tons')
        self.data['Car_Category'] = self.data['Car_Category'].fillna(replace_value)
        self.test['Car_Category'] = self.test['Car_Category'].fillna(replace_value)
        self.data['Car_Category'] = self.data['Car_Category'].apply(lambda x: x.title().replace(" ",""))
        self.test['Car_Category'] = self.test['Car_Category'].apply(lambda x: x.title().replace(" ",""))

    def clean_car_product(self, replace_value):
        self.data['ProductName'] = self.data['ProductName'].fillna(replace_value)
        self.test['ProductName'] = self.test['ProductName'].fillna(replace_value)
        self.data['ProductName'] = self.data['ProductName'].apply(lambda x: x.title().replace(" ",""))
        self.test['ProductName'] = self.test['ProductName'].apply(lambda x: x.title().replace(" ",""))

    def clean_date_col(self):
        st_day = self.preproc_args['missing']['policy_start_day']
        st_month = self.preproc_args['missing']['policy_start_month']
        st_year = self.preproc_args['missing']['policy_start_year']
        en_day = self.preproc_args['missing']['policy_end_day']
        en_month = self.preproc_args['missing']['policy_end_month']
        en_year = self.preproc_args['missing']['policy_end_year']
        tr_day = self.preproc_args['missing']['transaction_day']
        tr_month = self.preproc_args['missing']['transaction_month']
        tr_year = self.preproc_args['missing']['transaction_year']

        self.data['Policy_Start_Day'] = self.data['Policy_Start_Day'].fillna(st_day)
        self.test['Policy_Start_Day'] = self.test['Policy_Start_Day'].fillna(st_day)
        self.data['Policy_Start_Month'] = self.data['Policy_Start_Month'].fillna(st_month)
        self.test['Policy_Start_Month'] = self.test['Policy_Start_Month'].fillna(st_month)
        self.data['Policy_Start_Year'] = self.data['Policy_Start_Year'].fillna(st_year)
        self.test['Policy_Start_Year'] = self.test['Policy_Start_Year'].fillna(st_year)

        self.data['Policy_End_Day'] = self.data['Policy_End_Day'].fillna(en_day)
        self.test['Policy_End_Day'] = self.test['Policy_End_Day'].fillna(en_day)
        self.data['Policy_End_Month'] = self.data['Policy_End_Month'].fillna(en_month)
        self.test['Policy_End_Month'] = self.test['Policy_End_Month'].fillna(en_month)
        self.data['Policy_End_Year'] = self.data['Policy_End_Year'].fillna(en_year)
        self.test['Policy_End_Year'] = self.test['Policy_End_Year'].fillna(en_year)

        self.data['Transaction_Day'] = self.data['Transaction_Day'].fillna(en_day)
        self.test['Transaction_Day'] = self.test['Transaction_Day'].fillna(en_day)
        self.data['Transaction_Month'] = self.data['Transaction_Month'].fillna(en_month)
        self.test['Transaction_Month'] = self.test['Transaction_Month'].fillna(en_month)
        self.data['Transaction_Year'] = self.data['Transaction_Year'].fillna(en_year)
        self.test['Transaction_Year'] = self.test['Transaction_Year'].fillna(en_year)


    def remake_date_cols(self):
        compute = lambda x: break_date(extract_date(x))
        series_data = self.data['Policy Start Date']
        series_test = self.test['Policy Start Date']
        self.data[['Policy_Start_Day','Policy_Start_Month','Policy_Start_Year']] = pd.DataFrame(series_data.apply(
            compute).to_list())
        self.test[['Policy_Start_Day','Policy_Start_Month','Policy_Start_Year']] = pd.DataFrame(series_test.apply(
            compute).to_list())
        series_data = self.data['Policy End Date']
        series_test = self.test['Policy End Date']
        self.data[['Policy_End_Day','Policy_End_Month','Policy_End_Year']] = pd.DataFrame(series_data.apply(
            compute).to_list())
        self.test[['Policy_End_Day','Policy_End_Month','Policy_End_Year']] = pd.DataFrame(series_test.apply(
            compute).to_list())
        series_data = self.data['First Transaction Date']
        serses_test = self.test['First Transaction Date']
        self.data[['Transaction_Day','Transaction_Month','Transaction_Year']] = pd.DataFrame(series_data.apply(
            compute).to_list())
        self.test[['Transaction_Day','Transaction_Month','Transaction_Year']] = pd.DataFrame(series_test.apply(
            compute).to_list())

    def get_state_mapping(self, row):
        if is_null(row['LGA_Name']) and not(is_null(row['State'])):
            row['LGA_Name'] = self.state_map[self.state_map['State'] == row['State']].iloc[0]['LGA']
        elif not(is_null(row['LGA_Name'])) and is_null(row['State']):
            row['State'] = self.state_map[self.state_map['LGA'] == row['LGA_Name']].iloc[0]['State']
        elif is_null(row['LGA_Name']) and is_null(row['State']):
            row['State'] = self.preproc_args['missing']['state']
            row['LGA_Name'] = self.state_map[self.state_map['State'] == row['State']].iloc[0]['LGA']
        return row

    def clean_special_chars(self, row):
        temp_val = row['LGA_Name']
        temp_val = re.sub(r'[-/_]+',' ', temp_val)
        row['LGA_Name'] = temp_val.title().replace(' ','')
        temp_val = row['State']
        temp_val = re.sub(r'[-/_]+',' ', temp_val)
        row['State'] = temp_val.title().replace(' ','')
        return row

    def remake_states_lga(self):
        self.data['State'] = self.data['State'].replace('N-A',np.NaN)
        self.data['LGA_Name'] = self.data['LGA_Name'].replace('LGA',np.NaN)
        self.test['State'] = self.test['State'].replace('N-A',np.NaN)
        self.test['LGA_Name'] = self.test['LGA_Name'].replace('LGA',np.NaN)
        self.data = self.data.apply(self.get_state_mapping, axis = 1)
        self.test = self.test.apply(self.get_state_mapping, axis = 1)
        self.data = self.data.apply(self.clean_special_chars, axis = 1)
        self.test = self.test.apply(self.clean_special_chars, axis = 1)

    def remake_nopol_col(self):
        self.data['No_Pol'] = self.data['No_Pol'].astype(str)
        self.test['No_Pol'] = self.test['No_Pol'].astype(str)

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

    # Legacy Methods

    def clean_LGA_col(self, replace_value):
        self.data['LGA_Name'] = self.data['LGA_Name'].fillna(replace_value)
        self.test['LGA_Name'] = self.test['LGA_Name'].fillna(replace_value)

    def clean_state_col(self, replace_value):
        self.data['State'] = self.data['State'].fillna(replace_value)
        self.test['State'] = self.test['State'].fillna(replace_value)

