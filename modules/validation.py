from modules.utils import get_validation_params
import numpy as np

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


