from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import os

from MlClasses.PerformanceTests import classificationReport,rocCurve,compareTrainTest
from MlClasses.Config import Config

from sklearn.model_selection import cross_val_score

class Bdt(object):
    '''Take some data split into test and train sets and train a bdt on it'''
    def __init__(self,data,output=None):
        self.data = data
        self.output = output
        self.config=Config(output=output)

        self.accuracy=None
        self.crossValResults=None

    def setup(self,dtArgs={},bdtArgs={}):

        #Uses TMVA parameters as default
        if len(dtArgs)==0: 
            dtArgs['max_depth']=3
            #dtArgs['max_depth']=5
            dtArgs['min_samples_leaf']=0.05

        if len(bdtArgs)==0:
            bdtArgs['algorithm']='SAMME'
            bdtArgs['n_estimators']=800
            bdtArgs['learning_rate']=0.5
            #bdtArgs['learning_rate']=1.0

        self.dt = DecisionTreeClassifier(**dtArgs)
        self.bdt = AdaBoostClassifier(self.dt,**bdtArgs)

        self.config.addToConfig('nDevEvents',len(self.data.y_dev.index))
        self.config.addToConfig('nTrainEvents',len(self.data.y_train.index))
        self.config.addToConfig('nTestEvents',len(self.data.y_test.index))
        self.config.addToConfig('DT arguments',dtArgs)
        self.config.addToConfig('BDT arguments',bdtArgs)

    def fit(self):

        self.bdt.fit(self.data.X_train, self.data.y_train)

    def crossValidation(self,kfolds=3,n_jobs=4):
        '''K-means cross validation'''
        self.crossValResults = cross_val_score(self.bdt, self.data.X_dev, self.data.y_dev,scoring='accuracy',n_jobs=n_jobs,cv=kfolds)

        self.config.addLine('CrossValidation')
        self.config.addToConfig('kfolds',kfolds)
        self.config.addLine('')


    def classificationReport(self):
        if not os.path.exists(self.output): os.makedirs(self.output)
        f=open(os.path.join(self.output,'classificationReport.txt'),'w')
        f.write( 'Performance on test set:')
        classificationReport(self.bdt.predict(self.data.X_test),self.bdt.decision_function(self.data.X_test),self.data.y_test,f)

        f.write( '\n' )
        f.write('Performance on training set:')
        classificationReport(self.bdt.predict(self.data.X_train),self.bdt.decision_function(self.data.X_train),self.data.y_train,f)

        if self.crossValResults is not None:
            f.write( '\n\nCross Validation\n')
            f.write("Cross val results: %.2f%% (%.2f%%)" % (self.crossValResults.mean()*100, self.crossValResults.std()*100))
        
    def rocCurve(self):
        rocCurve(self.bdt.decision_function(self.data.X_test),self.data.y_test,output=self.output)
        rocCurve(self.bdt.decision_function(self.data.X_train),self.data.y_train,output=self.output,append='_train')

    def compareTrainTest(self):
        compareTrainTest(self.bdt.decision_function,self.data.X_train,self.data.y_train,\
                self.data.X_test,self.data.y_test,self.output)

    def diagnostics(self):
        self.classificationReport()
        self.rocCurve()
        self.compareTrainTest()

    def plotDiscriminator(self):
        plotDiscriminator(self.bdt,self.data.X_test,self.data.y_test, self.output)

    def testPrediction(self):
        return self.bdt.decision_function(self.data.X_test)

    def getAccuracy(self):
        if not self.accuracy:
            self.accuracy = self.bdt.score(self.data.X_test,self.data.y_test)
        return self.accuracy


