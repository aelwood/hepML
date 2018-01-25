import matplotlib.pyplot as plt
import os

from keras.utils import plot_model
from keras.wrappers.scikit_learn import KerasClassifier

from MlClasses.PerformanceTests import rocCurve,compareTrainTest,classificationReport,learningCurve,plotPredVsTruth
from MlClasses.Config import Config

from MlFunctions.DnnFunctions import createDenseModel

#For cross validation and HP tuning
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


class Dnn(object):

    def __init__(self,data,output=None,doRegression=False):
        self.data = data
        self.output = output
        self.config=Config(output=output)

        self.score=None
        self.crossValResults=None

        self.defaultParams = {}

        self.doRegression = doRegression

        if self.doRegression:
            self.scoreType = 'mean_squared_error'
        else:
            self.scoreType = 'acc'

    def setup(self,hiddenLayers=[1.0],dropOut=None,l2Regularization=None):

        '''Setup the neural net. Input a list of hiddenlayers
        if you fill float takes as fraction of inputs+outputs
        if you fill int takes as number. 
        E.g.:  hiddenLayers=[0.66,20] 
        has two hidden layers, one with 2/3 of input plus output 
        neurons, second with 20 neurons.
        All the layers are fully connected (dense)
        ''' 

        inputSize=len(self.data.X_train.columns)

        #Find number of unique outputs
        if self.doRegression: 
            outputSize = 1
        else: 
            outputSize = len(self.data.y_train.unique())

        self.defaultParams = dict(
            inputSize=inputSize,outputSize=outputSize,
            hiddenLayers=hiddenLayers,dropOut=dropOut,
            l2Regularization=l2Regularization,
            activation='relu',optimizer='adam',
            doRegression=self.doRegression
            )
        self.model=createDenseModel(**self.defaultParams)

        #Add stuff to the config
        self.config.addToConfig('nEvalEvents',len(self.data.y_eval.index))
        self.config.addToConfig('nDevEvents',len(self.data.y_dev.index))
        self.config.addToConfig('nTrainEvents',len(self.data.y_train.index))
        self.config.addToConfig('nTestEvents',len(self.data.y_test.index))
        for name,val in self.defaultParams.iteritems():
            self.config.addToConfig(name,val)

    def fit(self,epochs=20,batch_size=32,**kwargs):
        '''Fit with training set and validate with test set'''

        #Fit the model and save the history for diagnostics
        #additionally pass the testing data for further diagnostic results
        self.history = self.model.fit(self.data.X_train.as_matrix(), self.data.y_train.as_matrix(), 
                validation_data=(self.data.X_test.as_matrix(),self.data.y_test.as_matrix()),
                epochs=epochs, batch_size=batch_size,**kwargs)

        #Add stuff to the config
        self.config.addLine('Test train split')
        self.config.addToConfig('epochs',epochs)
        self.config.addToConfig('batch_size',batch_size)
        self.config.addToConfig('extra',kwargs)
        self.config.addLine('')

    def crossValidation(self,kfolds=3,epochs=20,batch_size=32,n_jobs=4):
        '''K-means cross validation with data standardisation'''
        #Reference: https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/

        #Have to be careful with this and redo the standardisation
        #Do this on the development set, save the eval set for after HP tuning

        #Check the development set isn't standardised
        if self.data.standardisedDev:
            self.data.unStandardise(justDev=True)

        #Use a pipeline in sklearn to carry out standardisation just on the training set
        estimators = []
        estimators.append(('standardize', preprocessing.StandardScaler()))
        estimators.append(('mlp', KerasClassifier(build_fn=createDenseModel, 
            epochs=epochs, batch_size=batch_size,verbose=0, **self.defaultParams))) #verbose=0
        pipeline = Pipeline(estimators)

        #Define the kfold parameters and run the cross validation
        kfold = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=43)
        self.crossValResults = cross_val_score(pipeline, self.data.X_dev.as_matrix(), self.data.y_dev.as_matrix(), cv=kfold,n_jobs=n_jobs)
        print "Cross val results: %.2f%% (%.2f%%)" % (self.crossValResults.mean()*100, self.crossValResults.std()*100)

        #Save the config
        self.config.addLine('CrossValidation')
        self.config.addToConfig('kfolds',kfolds)
        self.config.addToConfig('epochs',epochs)
        self.config.addToConfig('batch_size',batch_size)
        self.config.addLine('')

    def gridSearch(self,param_grid,kfolds=3,epochs=20,batch_size=32,n_jobs=4):
        '''Implementation of the sklearn grid search for hyper parameter tuning, 
        making use of kfolds cross validation.
        Pass a dictionary of lists of parameters to test on. Choose number of cores
        to run on with n_jobs, -1 is all of them'''
        #Reference: https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

        #Check the development set isn't standardised
        if self.data.standardisedDev:
            self.data.unStandardise(justDev=True)
        
        #Use a pipeline in sklearn to carry out standardisation just on the training set
        estimators = []
        estimators.append(('standardize', preprocessing.StandardScaler()))
        estimators.append(('mlp', KerasClassifier(build_fn=createDenseModel, 
            epochs=epochs, batch_size=batch_size,verbose=0, **self.defaultParams)))
        pipeline = Pipeline(estimators)

        #Setup and fit the grid search, choosing cross validation params and N cores
        grid = GridSearchCV(estimator=pipeline,param_grid=param_grid,cv=kfolds,n_jobs=n_jobs)
        self.gridResult = grid.fit(self.data.X_dev.as_matrix(),self.data.y_dev.as_matrix())

        #Save the results
        if not os.path.exists(self.output): os.makedirs(self.output)
        outFile = open(os.path.join(self.output,'gridSearchResults.txt'),'w')
        outFile.write("Best: %f using %s \n\n" % (self.gridResult.best_score_, self.gridResult.best_params_))
        means = self.gridResult.cv_results_['mean_test_score']
        stds = self.gridResult.cv_results_['std_test_score']
        params = self.gridResult.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            outFile.write("%f (%f) with: %r\n" % (mean, stdev, param))
        outFile.close()

    def saveConfig(self):

        if not os.path.exists(self.output): os.makedirs(self.output)
        #plot_model(self.model, to_file=os.path.join(self.output,'model.ps'))
        self.config.saveConfig()

    def classificationReport(self,doEvalSet=False):

        if doEvalSet: #produce report for dev and eval sets instead
            X_test=self.data.X_eval
            y_test=self.data.y_eval
            X_train=self.data.X_dev
            y_train=self.data.y_dev
            append='_eval'
        else:
            X_test=self.data.X_test
            y_test=self.data.y_test
            X_train=self.data.X_train
            y_train=self.data.y_train
            append=''

        if not os.path.exists(self.output): os.makedirs(self.output)
        f=open(os.path.join(self.output,'classificationReport'+append+'.txt'),'w')

        f.write( 'Performance on test set:\n')
        report = self.model.evaluate(X_test.as_matrix(), y_test.as_matrix(), batch_size=32)
        self.score=report[1]

        if self.doRegression:
            f.write( '\n\nDNN Loss, Mean Squared Error:\n')
        else:
            classificationReport(self.model.predict_classes(X_test.as_matrix()),self.model.predict(X_test.as_matrix()),y_test,f)
            f.write( '\n\nDNN Loss, Accuracy:\n')
        f.write(str(report)) 

        f.write( '\n\nPerformance on train set:\n')
        report = self.model.evaluate(X_train.as_matrix(), y_train.as_matrix(), batch_size=32)
        if self.doRegression:
            f.write( '\n\nDNN Loss, Mean Squared Error:\n')
        else:
            classificationReport(self.model.predict_classes(X_train.as_matrix()),self.model.predict(X_train.as_matrix()),y_train,f)
            f.write( '\n\nDNN Loss, Accuracy:\n')
        f.write(str(report))

        if self.crossValResults is not None:
            f.write( '\n\nCross Validation\n')
            f.write("Cross val results: %.2f%% (%.2f%%)" % (self.crossValResults.mean()*100, self.crossValResults.std()*100))

    def rocCurve(self,doEvalSet=False):

        if self.doRegression:
            print 'Cannot make ROC curve for a regression problem, skipping'
            return None

        if doEvalSet:
            rocCurve(self.model.predict(self.data.X_eval.as_matrix()), self.data.y_eval,self.output,append='_eval')
            rocCurve(self.model.predict(self.data.X_train.as_matrix()),self.data.y_train,self.output,append='_dev')
        else:
            rocCurve(self.model.predict(self.data.X_test.as_matrix()), self.data.y_test,self.output)
            rocCurve(self.model.predict(self.data.X_train.as_matrix()),self.data.y_train,self.output,append='_train')

    def compareTrainTest(self,doEvalSet=False):

        if doEvalSet: 
            compareTrainTest(self.model.predict,self.data.X_dev.as_matrix(),self.data.y_dev.as_matrix(),\
                self.data.X_eval.as_matrix(),self.data.y_eval.as_matrix(),self.output,append='_eval')
        else:
            compareTrainTest(self.model.predict,self.data.X_train.as_matrix(),self.data.y_train.as_matrix(),\
                self.data.X_test.as_matrix(),self.data.y_test.as_matrix(),self.output)

    def plotPredVsTruth(self,doEvalSet=False):

        if doEvalSet:
            plotPredVsTruth(self.model.predict(self.data.X_eval.as_matrix()), self.data.y_eval,self.output,append='_eval')
            plotPredVsTruth(self.model.predict(self.data.X_train.as_matrix()),self.data.y_train,self.output,append='_dev')
        else:
            plotPredVsTruth(self.model.predict(self.data.X_test.as_matrix()), self.data.y_test,self.output)
            plotPredVsTruth(self.model.predict(self.data.X_train.as_matrix()),self.data.y_train,self.output,append='_train')


    def plotHistory(self):

        if not os.path.exists(self.output): os.makedirs(self.output)

        #Plot the score over the epochs
        plt.plot(self.history.history[self.scoreType],label='train')
        plt.plot(self.history.history['val_'+self.scoreType],label='test')
        plt.title('model '+self.scoreType)
        plt.ylabel(self.scoreType)
        plt.xlabel('epoch')
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(self.output,'scoreEvolution.pdf'))
        plt.clf()

        #Plot the loss over the epochs
        plt.plot(self.history.history['loss'],label='train')
        plt.plot(self.history.history['val_loss'],label='test')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(self.output,'lossEvolution.pdf'))
        plt.clf()

    def learningCurve(self,epochs=20,batch_size=32,kfolds=3,n_jobs=1,scoring=None):
        model=KerasClassifier(build_fn=createDenseModel, 
            epochs=epochs, batch_size=batch_size,verbose=0, **self.defaultParams)   
        learningCurve(model,self.data.X_dev.as_matrix(),self.data.y_dev.as_matrix(),self.output,cv=kfolds,n_jobs=n_jobs,scoring=scoring)

    def diagnostics(self,doEvalSet=False):

        self.saveConfig()
        self.classificationReport(doEvalSet=doEvalSet)
        if not self.doRegression: 
            self.rocCurve(doEvalSet=doEvalSet)
            self.compareTrainTest(doEvalSet=doEvalSet)
        else:
            self.plotPredVsTruth(doEvalSet=doEvalSet)
        self.plotHistory()

    def testPrediction(self):
        return self.model.predict(self.data.X_test.as_matrix())

    def getAccuracy(self):

        if not self.score:
            self.score = self.model.evaluate(self.data.X_test.as_matrix(), self.data.y_test.as_matrix(), batch_size=32)[1]

        return self.score

