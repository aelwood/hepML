import matplotlib.pyplot as plt
import os

from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.utils import plot_model
from keras.wrappers.scikit_learn import KerasClassifier

from MlClasses.PerformanceTests import rocCurve,compareTrainTest,classificationReport
from MlClasses.Config import Config

#For cross validation
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


def findLayerSize(layer,refSize):

    if isinstance(layer, float):
        return int(layer*refSize)
    elif isinstance(layer, int):
        return layer
    else:
        print 'WARNING: layer must be int or float'
        return None

class Dnn(object):

    def __init__(self,data,output=None):
        self.data = data
        self.output = output
        self.config=Config(output=output)

        self.accuracy=None
        self.crossValResults=None

        self.defaultParams = {}

    #Add a function to create a model which can be called
    #when using sklearn crossvalidation and hyperparameter
    #tuning libraries
    @staticmethod
    def createModel(inputSize=None,outputSize=None,hiddenLayers=[1.0],dropOut=None,activation='relu',optimizer='adam'):

        #check inputs are ok
        assert inputSize and outputSize, 'Must provide non-zero input and output sizes'
        assert len(hiddenLayers)>=1, 'Need at least one hidden layer'

        refSize=inputSize+outputSize

        model = Sequential()

        #Add the first layer, taking the inputs
        model.add(Dense(units=findLayerSize(hiddenLayers[0],refSize), 
            activation=activation, input_dim=inputSize,name='input'))

        if dropOut: model.add(Dropout(dropOut))

        #Add the extra hidden layers
        for layer in hiddenLayers[1:]:
            model.add(Dense(units=findLayerSize(hiddenLayers[0],refSize), 
                activation=activation))

            if dropOut: model.add(Dropout(dropOut))

        #Add the output layer and choose the type of loss function
        #Choose the loss function based on whether it's binary or not
        if outputSize==2: 
            #It's better to choose a sigmoid function and one output layer for binary
            # This is a special case of n>2 classification
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else: 
            #Softmax forces the outputs to sum to 1 so the score on each node
            # can be interpreted as the probability of getting each class
            model.add(Dense(outputSize, activation='softmax'))
            loss = 'categorical_crossentropy'

        #After the layers are added compile the model
        model.compile(loss=loss,
            optimizer=optimizer,metrics=['accuracy'])

        return model

    def setup(self,hiddenLayers=[1.0],dropOut=None):

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
        outputSize = len(self.data.y_train.unique())

        self.model=self.createModel(
            inputSize=inputSize,outputSize=outputSize,
            hiddenLayers=hiddenLayers,dropOut=dropOut,
            activation='relu',optimizer='adam'
            )
        self.defaultParams = dict(
            inputSize=inputSize,outputSize=outputSize,
            hiddenLayers=hiddenLayers,dropOut=dropOut,
            activation='relu',optimizer='adam'
            )

        #Add stuff to the config
        self.config.addToConfig('inputSize',inputSize)
        self.config.addToConfig('outputSize',outputSize)
        self.config.addToConfig('hiddenLayers',hiddenLayers)
        self.config.addToConfig('dropOut',dropOut)

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

    def crossValidation(self,kfolds=10,epochs=20,batch_size=32):
        '''K-means cross validation with data standardisation'''
        #Have to be careful with this and redo the standardisation
        #Do this on the development set, save the eval set for after HP tuning

        #Check the development set isn't standardised
        if self.data.standardisedDev:
            self.data.unStandardise(justDev=True)

        #Use a pipeline in sklearn to carry out standardisation just on the training set
        estimators = []
        estimators.append(('standardize', preprocessing.StandardScaler()))
        estimators.append(('mlp', KerasClassifier(build_fn=self.createModel, 
            epochs=epochs, batch_size=batch_size,verbose=0, **self.defaultParams))) #verbose=0
        pipeline = Pipeline(estimators)

        #Define the kfold parameters and run the cross validation
        kfold = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=43)
        self.crossValResults = cross_val_score(pipeline, self.data.X_dev.as_matrix(), self.data.y_dev.as_matrix(), cv=kfold)
        print "Cross val results: %.2f%% (%.2f%%)" % (self.crossValResults.mean()*100, self.crossValResults.std()*100)

        #Save the config
        self.config.addLine('CrossValidation')
        self.config.addToConfig('kfolds',kfolds)
        self.config.addToConfig('epochs',epochs)
        self.config.addToConfig('batch_size',batch_size)
        self.config.addLine('')

    def saveConfig(self):

        if not os.path.exists(self.output): os.makedirs(self.output)
        plot_model(self.model, to_file=os.path.join(self.output,'model.ps'))
        self.config.saveConfig()

    def classificationReport(self):

        if not os.path.exists(self.output): os.makedirs(self.output)
        f=open(os.path.join(self.output,'classificationReport.txt'),'w')
        f.write( 'Performance on test set:\n')
        report = self.model.evaluate(self.data.X_test.as_matrix(), self.data.y_test.as_matrix(), batch_size=32)
        self.accuracy=report[1]
        classificationReport(self.model.predict_classes(self.data.X_test.as_matrix()),self.model.predict(self.data.X_test.as_matrix()),self.data.y_test,f)
        f.write( '\n\nDNN Loss, Accuracy:\n')
        f.write(str(report)) 

        f.write( '\n\nPerformance on train set:\n')
        report = self.model.evaluate(self.data.X_train.as_matrix(), self.data.y_train.as_matrix(), batch_size=32)
        classificationReport(self.model.predict_classes(self.data.X_train.as_matrix()),self.model.predict(self.data.X_train.as_matrix()),self.data.y_train,f)
        f.write( '\n\nDNN Loss, Accuracy:\n')
        f.write(str(report))

        if self.crossValResults is not None:
            f.write( '\n\nCross Validation\n')
            f.write("Cross val results: %.2f%% (%.2f%%)" % (self.crossValResults.mean()*100, self.crossValResults.std()*100))

    def rocCurve(self):

        rocCurve(self.model.predict(self.data.X_test.as_matrix()), self.data.y_test,self.output)
        rocCurve(self.model.predict(self.data.X_train.as_matrix()),self.data.y_train,self.output,append='_train')

    def compareTrainTest(self):
        compareTrainTest(self.model.predict,self.data.X_train.as_matrix(),self.data.y_train.as_matrix(),\
                self.data.X_test.as_matrix(),self.data.y_test.as_matrix(),self.output)

    def plotHistory(self):

        if not os.path.exists(self.output): os.makedirs(self.output)

        #Plot the accuracy over the epochs
        plt.plot(self.history.history['acc'],label='train')
        plt.plot(self.history.history['val_acc'],label='test')
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(self.output,'accuracyEvolution.pdf'))
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

    def diagnostics(self):

        self.saveConfig()
        self.classificationReport()
        self.rocCurve()
        self.compareTrainTest()
        self.plotHistory()

    def testPrediction(self):
        return self.model.predict(self.data.X_test.as_matrix())

    def getAccuracy(self):

        if not self.accuracy:
            self.accuracy = self.model.evaluate(self.data.X_test.as_matrix(), self.data.y_test.as_matrix(), batch_size=32)[1]

        return self.accuracy
        
