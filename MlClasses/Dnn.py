import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import pickle
import random

from keras.utils import plot_model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model

from lime import lime_tabular

from MlClasses.PerformanceTests import rocCurve,compareTrainTest,classificationReport,learningCurve,plotPredVsTruth
from MlClasses.Config import Config
from MlFunctions.DnnFunctions import createDenseModel

from pandasPlotting.Plotter import Plotter

#For cross validation and HP tuning
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from asimovErrors import Z,eZ


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
            self.scoreTypes = ['mean_squared_error']
        else:
            self.scoreTypes = ['acc']

    def setup(self,hiddenLayers=[1.0],dropOut=None,l2Regularization=None,loss=None,extraMetrics=[]):

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
        self.model=createDenseModel(loss=loss,extraMetrics=extraMetrics,**self.defaultParams)
        for em in extraMetrics:
            if isinstance(em,str): 
                self.scoreTypes.append(em)
            else:
                self.scoreTypes.append(em.__name__)

        #Add stuff to the config
        self.config.addToConfig('Vars used',self.data.X.columns.values)
        self.config.addToConfig('nEvalEvents',len(self.data.y_eval.index))
        self.config.addToConfig('nDevEvents',len(self.data.y_dev.index))
        self.config.addToConfig('nTrainEvents',len(self.data.y_train.index))
        self.config.addToConfig('nTestEvents',len(self.data.y_test.index))
        for name,val in self.defaultParams.iteritems():
            self.config.addToConfig(name,val)

    def recompileModel(self,loss):
        '''Recompile the model if you want to change the loss for example'''
        self.model.compile(loss=loss,
            optimizer=self.model.optimizer,metrics=self.model.metrics)

    def fit(self,epochs=20,batch_size=32,**kwargs):
        '''Fit with training set and validate with test set'''

        #Fit the model and save the history for diagnostics
        #additionally pass the testing data for further diagnostic results
        self.history = self.model.fit(self.data.X_train.as_matrix(), self.data.y_train.as_matrix(), sample_weight=self.data.weights_train,
                validation_data=(self.data.X_test.as_matrix(),self.data.y_test.as_matrix(),self.data.weights_test),
                epochs=epochs, batch_size=batch_size,**kwargs)

        #Add stuff to the config
        self.config.addLine('Test train split')
        self.config.addToConfig('epochs',epochs)
        self.config.addToConfig('batch_size',batch_size)
        self.config.addToConfig('extra',kwargs)
        self.config.addLine('')

    def save(self):
        '''Save the model to reuse'''
        output=os.path.join(self.output,'model')
        if not os.path.exists(output): os.makedirs(output)

        #Save the trained model with weights etc
        self.model.save(os.path.join(output,'model.h5'))

        #Save the variables used in pickle format
        outF = open(os.path.join(output,'vars.pkl'),'w')
        pickle.dump(self.data.X.columns.values,outF)
        outF.close()

    def load(self,pathToModel,custom_objects=None):
        '''Load a previously saved model (in h5 format)'''
        self.model = load_model(pathToModel)

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

    def classificationReport(self,doEvalSet=False,batchSize=32):

        if doEvalSet: #produce report for dev and eval sets instead
            X_test=self.data.X_eval
            y_test=self.data.y_eval
            weights_test=self.data.weights_eval
            X_train=self.data.X_dev
            y_train=self.data.y_dev
            weights_train=self.data.weights_dev
            append='_eval'
        else:
            X_test=self.data.X_test
            y_test=self.data.y_test
            weights_test=self.data.weights_test
            X_train=self.data.X_train
            y_train=self.data.y_train
            weights_train=self.data.weights_train
            append=''

        if not os.path.exists(self.output): os.makedirs(self.output)
        f=open(os.path.join(self.output,'classificationReport'+append+'.txt'),'w')

        f.write( 'Performance on test set:\n')
        report = self.model.evaluate(X_test.as_matrix(), y_test.as_matrix(), sample_weight=weights_test, batch_size=batchSize)
        self.score=report

        if self.doRegression:
            f.write( '\n\nDNN Loss, Mean Squared Error:\n')
        else:
            classificationReport(self.model.predict_classes(X_test.as_matrix()),self.model.predict(X_test.as_matrix()),y_test,f)
            f.write( '\n\nDNN Loss, Accuracy, Significance:\n')
        f.write(str(report)) 

        f.write( '\n\nPerformance on train set:\n')
        report = self.model.evaluate(X_train.as_matrix(), y_train.as_matrix(), sample_weight=weights_train, batch_size=batchSize)
        if self.doRegression:
            f.write( '\n\nDNN Loss, Mean Squared Error:\n')
        else:
            classificationReport(self.model.predict_classes(X_train.as_matrix()),self.model.predict(X_train.as_matrix()),y_train,f)
            f.write( '\n\nDNN Loss, Accuracy, Significance:\n')
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
        for scoreType in self.scoreTypes:
            plt.plot(self.history.history[scoreType],label='train')
            plt.plot(self.history.history['val_'+scoreType],label='test')
            plt.title('model '+scoreType)
            plt.ylabel(scoreType)
            plt.xlabel('epoch')
            #do log if a big fractional difference
            if max(self.history.history[scoreType])>20*min(self.history.history[scoreType]) and min(self.history.history[scoreType])>0:
                plt.yscale('log')
            plt.legend(loc='upper left')
            plt.savefig(os.path.join(self.output,scoreType+'Evolution.pdf'))
            plt.clf()

        #Plot the loss over the epochs
        plt.plot(self.history.history['loss'],label='train')
        plt.plot(self.history.history['val_loss'],label='test')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        if max(self.history.history['loss'])>20*min(self.history.history['loss']) and min(self.history.history['loss'])>0:
            plt.yscale('log')
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(self.output,'lossEvolution.pdf'))
        plt.clf()

    def learningCurve(self,epochs=20,batch_size=32,kfolds=3,n_jobs=1,scoring=None):
        model=KerasClassifier(build_fn=createDenseModel, 
            epochs=epochs, batch_size=batch_size,verbose=0, **self.defaultParams)   
        learningCurve(model,self.data.X_dev.as_matrix(),self.data.y_dev.as_matrix(),self.output,cv=kfolds,n_jobs=n_jobs,scoring=scoring)

    def explainPredictions(self):
        '''Use LIME (https://github.com/marcotcr/lime) to give local explanations for the predictions of certain points'''

        explainer = lime_tabular.LimeTabularExplainer(training_data=self.data.X_train.values,  # training data
                                                   mode='classification',
                                                   feature_names=list(self.data.X_train),   # names of all features (regardless of type)
                                                   class_names=['background', 'signal'],            # class names
                                                   #class_names=[0,1],            # class names
                                                   discretize_continuous=True,
                                                   categorical_features= None,
                                                   categorical_names=None,
                                                   )

        def predict_fn_keras(x): 
            if x.ndim>=2:
                pred=self.model.predict(x,batch_size=1)
            else:
                pred=self.model.predict(x.reshape(1,x.shape[-1]),batch_size=1)
            return np.concatenate((1.-pred,pred),axis=1)

        for i in range(0,10):
            #len(self.data.X_test)
            exp = explainer.explain_instance(data_row=self.data.X_test.values[random.randint(0,len(self.data.X_test)-1)],   # 2d numpy array, corresponding to a row
                                     predict_fn=predict_fn_keras,  # classifier prediction probability function, 
                                     labels=[1,],               # iterable with labels to be explained.
                                     num_features=self.data.X_train.shape[1],      # maximum number of features present in explanation
                                     #top_labels=0,                # explanations for the K labels with highest prediction probabilities,
                                     #num_samples=2000,            # size of the neighborhood to learn the linear model
                                     #distance_metric='euclidean'  # the distance metric to use for weights.
                                     )

            out = os.path.join(self.output,'explanations')
            if not os.path.exists(out): os.makedirs(out)
            exp.save_to_file(os.path.join(out,'explanation'+str(i)+'.html'))
           # print exp.as_pyplot_figure()
            exp.as_pyplot_figure().savefig(os.path.join(out,'explanation'+str(i)+'.png'))
        pass

    def diagnostics(self,doEvalSet=False,batchSize=32,subDir=None):

        if subDir:
            oldOutput = self.output
            self.output=os.path.join(self.output,subDir)

        self.saveConfig()
        self.classificationReport(doEvalSet=doEvalSet,batchSize=batchSize)
        if not self.doRegression: 
            self.rocCurve(doEvalSet=doEvalSet)
            self.compareTrainTest(doEvalSet=doEvalSet)
        else:
            self.plotPredVsTruth(doEvalSet=doEvalSet)
        self.plotHistory()

        if subDir:
            self.output=oldOutput

    def testPrediction(self):
        return self.model.predict(self.data.X_test.as_matrix())

    def getAccuracy(self,batchSize=32):

        if not self.score:
            self.score = self.model.evaluate(self.data.X_test.as_matrix(), self.data.y_test.as_matrix(), batch_size=batchSize)

        return self.score

    def makeHepPlots(self,expectedSignal,expectedBackground,systematics=[0.0001],makeHistograms=True,subDir=None,customPrediction=None):
        '''Plots intended for binary signal/background classification
        
            - Plots significance as a function of discriminator output
            - Plots the variables for signal and background given different classifications 
            - If reference variables are available in the data they will also be plotted 
            (to be implemented, MlData.referenceVars)
        '''

        if subDir:
            oldOutput = self.output
            self.output=os.path.join(self.output,subDir)

        if self.doRegression:
            print 'makeHepPlots not implemented for regression'
            return None

        names = self.data.X.columns.values

        #Then predict the results and save them
        if customPrediction is None:
            predictionsTest = self.testPrediction()
        else:
            predictionsTest = customPrediction

        dataTest=pd.DataFrame(self.data.scaler.inverse_transform(self.data.X_test),columns=names)

        #Add the predictions and truth to a data frame
        dataTest['truth']=self.data.y_test.as_matrix()
        dataTest['pred']=predictionsTest

        signalSizeTest = len(dataTest[dataTest.truth==1])
        bkgdSizeTest = len(dataTest[dataTest.truth==0])
        signalWeightTest = float(expectedSignal)/signalSizeTest
        bkgdWeightTest = float(expectedBackground)/bkgdSizeTest

        def applyWeight(row,sw,bw):
            if row.truth==1: return sw
            else: return bw

        dataTest['weight'] = dataTest.apply(lambda row: applyWeight(row,signalWeightTest,bkgdWeightTest), axis=1)

        #save it for messing about
        #dataTest.to_pickle('dataTestSigLoss.pkl')

        #Produce a cumulative histogram of signal and background (truth) as a function of score
        #Plot it with a log scTrue

        h1=plt.hist(dataTest[dataTest.truth==0]['pred'],weights=dataTest[dataTest.truth==0]['weight'],bins=5000,color='b',alpha=0.8,label='background',cumulative=-1)
        h2=plt.hist(dataTest[dataTest.truth==1]['pred'],weights=dataTest[dataTest.truth==1]['weight'],bins=5000,color='r',alpha=0.8,label='signal',cumulative=-1)
        plt.yscale('log')
        plt.ylabel('Cumulative event counts / 0.02')
        plt.xlabel('Classifier output')
        plt.legend()
 
        plt.savefig(os.path.join(self.output,'cumulativeWeightedDiscriminator.pdf'))
        plt.clf()

        #From the cumulative histograms plot s/b and s/sqrt(s+b)

        s=h2[0]
        b=h1[0]

        plt.plot((h1[1][:-1]+h1[1][1:])/2,s/b)
        plt.title('sig/bkgd on test set')
        plt.savefig(os.path.join(self.output,'sigDivBkgdDiscriminator.pdf'))
        plt.clf()

        plt.plot((h1[1][:-1]+h1[1][1:])/2,s/np.sqrt(s+b))
        plt.title('sig/sqrt(sig+bkgd) on test set, best is '+str(max(s/np.sqrt(s+b))))
        plt.savefig(os.path.join(self.output,'sensitivityDiscriminator.pdf'))
        plt.clf()

        for systematic in systematics:
            # sigB=systematic*b
            #
            # toPlot=np.sqrt(2*( (s+b) * np.log( (s+b)*(b+sigB*sigB)/(b*b+(s+b)*sigB*sigB) ) - b*b*np.log( 1+sigB*sigB*s/(b*(b+sigB*sigB)) ) / (sigB*sigB) ))
            #plt.plot((h1[1][:-1]+h1[1][1:])/2,Z(s,b,systematic))
            toPlot = Z(s,b,systematic)
            plt.plot((h1[1][:-1]+h1[1][1:])/2,toPlot)
            es = signalWeightTest*np.sqrt(s/signalWeightTest)
            eb = bkgdWeightTest*np.sqrt(b/bkgdWeightTest)
            error=eZ(s,es,b,eb,systematic)
            # plt.plot((h1[1][:-1]+h1[1][1:])/2,toPlot-error)
            # plt.plot((h1[1][:-1]+h1[1][1:])/2,toPlot+error)
            plt.fill_between((h1[1][:-1]+h1[1][1:])/2,toPlot-error,toPlot+error,linewidth=0,alpha=0.6)
            maxIndex=np.argmax(toPlot)
            plt.title('Systematic '+str(systematic)+', s: '+str(round(s[maxIndex],1))+', b:'+str(round(b[maxIndex],1))+', best significance is '+str(round(toPlot[maxIndex],2))+' +/- '+str(round(error[maxIndex],2)))
            plt.xlabel('Cut on classifier score')
            plt.ylabel('Asimov estimate of significance')
            plt.savefig(os.path.join(self.output,'asimovDiscriminatorSyst'+str(systematic).replace('.','p')+'.pdf'))
            plt.clf()

            if makeHistograms: #Do this on the full set

                #Start with all the data and standardise it
                if self.data.standardised:
                    data = pd.DataFrame(self.data.scaler.transform(self.data.X))
                    #data = self.data.X.apply(self.data.scaler.transform)
                else:
                    data = self.data.X

                predictions= self.model.predict(data.as_matrix())
            
                #Now unstandardise
                if self.data.standardised:
                    data=pd.DataFrame(self.data.scaler.inverse_transform(data),columns=names)

                data['truth']=self.data.y.as_matrix()
                data['pred']=predictions

                signalSize = len(data[data.truth==1])
                bkgdSize = len(data[data.truth==0])
                signalWeight = float(expectedSignal)/signalSize
                bkgdWeight = float(expectedBackground)/bkgdSize

                data['weight'] = data.apply(lambda row: applyWeight(row,signalWeight,bkgdWeight), axis=1)

                #Plot all other interesting variables given classification
                p = Plotter(data,os.path.join(self.output,'allHistsSyst'+str(systematic).replace('.','p')))
                p1 = Plotter(data[data.pred>float(maxIndex)/len(toPlot)],os.path.join(self.output,'signalPredHistsSyst'+str(systematic).replace('.','p')))
                p2 = Plotter(data[data.pred<float(maxIndex)/len(toPlot)],os.path.join(self.output,'bkgdPredHistsSyst'+str(systematic).replace('.','p')))

                p.plotAllStackedHists1D('truth',weights='weight',log=True)
                p1.plotAllStackedHists1D('truth',weights='weight',log=True)
                p2.plotAllStackedHists1D('truth',weights='weight',log=True)

        if subDir:
            self.output=oldOutput
            
        pass

