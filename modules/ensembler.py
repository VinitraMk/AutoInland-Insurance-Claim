from sklearn.ensemble import VotingClassifier

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
        model = VotingClassifier(estimators=models, voting='hard')
        model.fit(self.X, self.y)
        return model

