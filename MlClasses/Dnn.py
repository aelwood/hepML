from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.utils import plot_model
from MlClasses.PerformanceTests import rocCurve,compareTrainTest
from MlClasses.Config import Config
import os

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

        self.model.fit(self.data.X_train.as_matrix(), self.data.y_train.as_matrix(), 
                epochs=epochs, batch_size=batch_size,**kwargs)

        #Add stuff to the config
        self.config.addToConfig('epochs',epochs)
        self.config.addToConfig('batch_size',batch_size)
        self.config.addToConfig('extra',kwargs)

    def saveConfig(self):

        plot_model(self.model, to_file=os.path.join(self.output,'model.ps'))
        self.config.saveConfig()

    def classificationReport(self):

        self.report = self.model.evaluate(self.data.X_test.as_matrix(), self.data.y_test.as_matrix(), batch_size=128)

        if not os.path.exists(self.output): os.makedirs(self.output)
        f=open(os.path.join(self.output,'classificationReport.txt'),'w')
        f.write( 'Performance on test set:')
        f.write(str(self.report))

    def rocCurve(self):

        rocCurve(self.model.predict(self.data.X_test.as_matrix()), self.data.y_test,self.output)
        rocCurve(self.model.predict(self.data.X_train.as_matrix()),self.data.y_train,self.output,append='_train')

    def compareTrainTest(self):
        compareTrainTest(self.model.predict,self.data.X_train.as_matrix(),self.data.y_train.as_matrix(),\
                self.data.X_test.as_matrix(),self.data.y_test.as_matrix(),self.output)

    def diagnostics(self):

        self.saveConfig()
        self.classificationReport()
        self.rocCurve()
        self.compareTrainTest()

    def testPrediction(self):
        return self.model.predict(self.data.X_test.as_matrix())

