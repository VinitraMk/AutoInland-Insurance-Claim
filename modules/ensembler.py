from sklearn.ensemble import VotingClassifier

from modules.utils import get_model_params

class Ensembler:
    ensembler_type = ''
    X = None
    y = None
    model = None


    def __init__(self, X, y, model, ensembler_type):
        self.X = X
        self.y = y
        self.model = model
        self.ensembler_type = ensembler_type

    def train_model(self, models):
        if self.ensembler_type == 'voting':
            return self.setup_voting_classifier(models)
        else:
            print('\nInvalid ensembler type :-|\n')
            exit()

    def setup_voting_classifier(self, models):
        params = get_model_params()
        model = VotingClassifier(estimators=models, voting=params['voting_type'])
        model.fit(self.X, self.y)
        return model

