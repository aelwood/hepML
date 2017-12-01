from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from MlClasses.PerformanceTests import classificationReport,rocCurve,compareTrainTest
import os

class Bdt(object):
    '''Take some data split into test and train sets and train a bdt on it'''
    def __init__(self,data,output=None):
        self.data = data
        self.output = output

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

    def fit(self):

        self.bdt.fit(self.data.X_train, self.data.y_train)

    def classificationReport(self):
        if not os.path.exists(self.output): os.makedirs(self.output)
        f=open(os.path.join(self.output,'classificationReport.txt'),'w')
        f.write( 'Performance on test set:')
        classificationReport(self.bdt,self.data.X_test,self.data.y_test,f)

        f.write( '\n' )
        f.write('Performance on training set:')
        classificationReport(self.bdt,self.data.X_train,self.data.y_train,f)
        
    def rocCurve(self):
        rocCurve(self.bdt,self.data.X_test,self.data.y_test,self.output)
        rocCurve(self.bdt,self.data.X_train,self.data.y_train,self.output,append='_train')

    def compareTrainTest(self):
        compareTrainTest(self.bdt,self.data.X_train,self.data.y_train,\
                self.data.X_test,self.data.y_test,self.output)

    def diagnostics(self):
        self.classificationReport()
        self.rocCurve()
        self.compareTrainTest()

    def plotDiscriminator(self):
        plotDiscriminator(self.bdt,self.data.X_test,self.data.y_test, self.output)




