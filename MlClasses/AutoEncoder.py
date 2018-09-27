import matplotlib.pyplot as plt
import os

from MlClasses.Dnn import Dnn
from MlClasses.Config import Config
from MlFunctions.DnnFunctions import createAutoencoderModel

class AutoEncoder(Dnn):

    def __init__(self,data,output=None):
        self.data = data
        self.output = output
        self.config=Config(output=output)

        self.score=None
        self.crossValResults=None

        self.defaultParams = {}

        self.scoreTypes = ['mean_squared_error']

    def setup(self,hiddenLayers=[30,10,30],dropOut=None,l2Regularization=None,loss='mean_squared_error',extraMetrics=[]):

        '''Setup the autoencoder. Input a list of hiddenlayers.
        ''' 

        inputSize=len(self.data.X_train.columns)


        self.defaultParams = dict(
            inputSize=inputSize,
            hiddenLayers=hiddenLayers,dropOut=dropOut,
            l2Regularization=l2Regularization,
            activation='relu',optimizer='adam'
            )

        self.model=createAutoencoderModel(loss=loss,extraMetrics=extraMetrics,**self.defaultParams)

        #Add the extra metrics
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

    def diagnostics(self, doEvalSet=False,batchSize=32,subDir=None):

        '''Custom diagnostics for the autoencoder'''

        if subDir:
            oldOutput = self.output
            self.output=os.path.join(self.output,subDir)

        self.saveConfig()
        self.plotHistory()

        if subDir:
            self.output=oldOutput

    def plotError(self,customPred=False,customData=None):

        data = self.data.X_test.as_matrix()
        append=''
        
        preds = self.model.predict(data) 

        mse = ((data - preds) ** 2).mean(axis=1)
        # print preds
        # print data
        # print mse


        if customPred: 
            data = customData
            append='WithCustom'
            preds = self.model.predict(data) 
            mse2 = ((data - preds) ** 2).mean(axis=1)
            maxRange=max(max(mse),max(mse2))
            plt.hist(mse2,bins=70,range=[0,maxRange],label='signal',alpha=0.7)
        else:
            maxRange=max(mse)

        plt.hist(mse,bins=70,range=[0,maxRange],label='bkgd',alpha=0.7)
        plt.yscale('log')
        plt.legend()

        if not os.path.exists(self.output): os.makedirs(self.output)
        plt.savefig(os.path.join(self.output,'error'+append+'.pdf'))
        plt.clf()
        

