from modules.utils import get_validation_params

import numpy as np
from sklearn.model_selection import train_test_split

class Validate:
    data = None
    train_X = None
    train_y = None
    val_X = None
    val_y = None
    validation_params = None

    def __init__(self, data):
        self.data = data
        self.validation_params = get_validation_params()

    def prepare_dataset(self):
        k = self.validation_params['k']
        val_len = int(self.data.shape[0] / k)
        val_indices = np.random.randint(self.data.shape[0], size=val_len)
        val_data = self.data.iloc[val_indices]
        train_data = self.data.drop(val_indices, axis=0)
        self.train_y = train_data['target']
        self.train_X = train_data.drop(columns=['target'], axis=1)
        self.val_y = val_data['target']
        self.val_X = val_data.drop(columns=['target'], axis=1)
        return self.train_X, self.train_y, self.val_X, self.val_y

    def prepare_full_dataset(self):
        params = get_validation_params()
        self.train_X = self.data.drop(columns=['target'])
        self.train_y = self.data['target']
        if not(params['split_data_for_training']):
            return self.train_X, self.train_y
        else:
            self.train_X, self.valid_X, self.train_y, self.valid_y = train_test_split(self.train_X,self.train_y,
                    test_size = params['data_split']['validate'], random_state = 42)
            return self.train_X, self.train_y, self.valid_X, self.valid_y


