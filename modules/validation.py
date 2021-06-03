from modules.utils import get_validation_params

class Validate:
    data = None
    train_X = None
    train_y = None
    val_X = None
    val_y = None
    validation_params = None

    def _init__(self, data):
        self.data = data
        self.validation_params = get_validation_params

    def apply_validation(self):
        k = self.validation_params['k']
        val_len = self.X.shape[0] / k

