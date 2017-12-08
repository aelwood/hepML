import matplotlib.pyplot as plt
import os

from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.utils import plot_model

from MlClasses.PerformanceTests import rocCurve,compareTrainTest,classificationReport
from MlClasses.Config import Config

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
        refSize=inputSize+outputSize

        self.model = Sequential()

        assert len(hiddenLayers)>=1, 'Need at least one hidden layer'

        #Add the first layer, taking the inputs
        self.model.add(Dense(units=findLayerSize(hiddenLayers[0],refSize), 
            activation='relu', input_dim=inputSize))

        if dropOut: self.model.add(Dropout(dropOut))

        #Add the extra hidden layers
        for layer in hiddenLayers[1:]:
            self.model.add(Dense(units=findLayerSize(hiddenLayers[0],refSize), 
                activation='relu'))

            if dropOut: self.model.add(Dropout(dropOut))

        #Add the output layer and choose the type of loss function
        #Choose the loss function based on whether it's binary or not
        if outputSize==2: 
            #It's better to choose a sigmoid function and one output layer for binary
            # This is a special case of n>2 classification
            self.model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else: 
            #Softmax forces the outputs to sum to 1 so the score on each node
            # can be interpreted as the probability of getting each class
            self.model.add(Dense(outputSize, activation='softmax'))
            loss = 'categorical_crossentropy'

        #After the layers are added compile the model
        self.model.compile(loss=loss,
            optimizer='adam',metrics=['accuracy'])

        #Add stuff to the config
        self.config.addToConfig('inputSize',inputSize)
        self.config.addToConfig('outputSize',outputSize)
        self.config.addToConfig('hiddenLayers',hiddenLayers)
        self.config.addToConfig('dropOut',dropOut)
        self.config.addToConfig('loss',loss)

    def fit(self,epochs=20,batch_size=32,**kwargs):

        #Fit the model and save the history for diagnostics
        #additionally pass the testing data for further diagnostic results
        self.history = self.model.fit(self.data.X_train.as_matrix(), self.data.y_train.as_matrix(), 
                validation_data=(self.data.X_test.as_matrix(),self.data.y_test.as_matrix()),
                epochs=epochs, batch_size=batch_size,**kwargs)

        #Add stuff to the config
        self.config.addToConfig('epochs',epochs)
        self.config.addToConfig('batch_size',batch_size)
        self.config.addToConfig('extra',kwargs)

    def saveConfig(self):

        if not os.path.exists(self.output): os.makedirs(self.output)
        plot_model(self.model, to_file=os.path.join(self.output,'model.ps'))
        self.config.saveConfig()

    def classificationReport(self):

        if not os.path.exists(self.output): os.makedirs(self.output)
        f=open(os.path.join(self.output,'classificationReport.txt'),'w')
        f.write( 'Performance on test set:\n')
        report = self.model.evaluate(self.data.X_test.as_matrix(), self.data.y_test.as_matrix(), batch_size=32)
        classificationReport(self.model.predict_classes(self.data.X_test.as_matrix()),self.model.predict(self.data.X_test.as_matrix()),self.data.y_test,f)
        f.write( '\nDNN Loss, Accuracy:\n')
        f.write(str(report)) 

        f.write( '\n\nPerformance on train set:\n')
        report = self.model.evaluate(self.data.X_train.as_matrix(), self.data.y_train.as_matrix(), batch_size=32)
        classificationReport(self.model.predict_classes(self.data.X_train.as_matrix()),self.model.predict(self.data.X_train.as_matrix()),self.data.y_train,f)
        f.write( '\nDNN Loss, Accuracy:\n')
        f.write(str(report))

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

