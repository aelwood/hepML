from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score

class Bdt(object):
    '''Take some data split into test and train sets and train a bdt on it'''
    def __init__(self,data):
        self.data = data

    def setup(self,dtArgs={},bdtArgs={}):

        #Uses TMVA parameters as default
        if len(dtArgs)==0: 
            dtArgs['max_depth']=3
            dtArgs['min_samples_leaf']=0.05*len(self.data.X_train)

        if len(bdtArgs)==0:
            bdtArgs['algorithm']='SAMME'
            bdtArgs['n_estimators']=800
            bdtArgs['learning_rate']=0.5

        self.dt = DecisionTreeClassifier(**dtArgs)
        self.bdt = AdaBoostClassifier(self.dt,**bdtArgs)

    def fit(self):
        self.bdt.fit(self.data.X_train, self.data.y_train)
