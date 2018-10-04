import matplotlib.pyplot as plt
import os
import numpy as np

from MlClasses.Dnn import Dnn
from MlClasses.Config import Config
from MlFunctions.DnnFunctions import createAutoencoderModel
from MlClasses.PerformanceTests import rocCurve

from asimovErrors import Z,eZ

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

    def plotError(self,customPred=False,customData=None,expectedCounts=None):

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
            mseCustom = ((data - preds) ** 2).mean(axis=1)
            maxRange=max(max(mse),max(mseCustom))
            plt.hist(mseCustom,bins=100,range=[0,maxRange],label='signal',alpha=0.5)
        else:
            maxRange=max(mse)

        plt.hist(mse,bins=100,range=[0,maxRange],label='bkgd',alpha=0.5)
        plt.yscale('log')
        plt.legend()

        if not os.path.exists(self.output): os.makedirs(self.output)
        plt.savefig(os.path.join(self.output,'error'+append+'.pdf'))
        plt.clf()

        #Also make a ROC curve to show separation power
        if customPred:
            y_pred = np.concatenate((mse,mseCustom))
            y_true= np.concatenate((np.zeros(len(mse)),np.ones(len(mseCustom))))
            rocCurve(y_pred,y_true,self.output)

            if expectedCounts is not None:
                assert len(expectedCounts)==2,'Need exactly two sets of expected counts'
                #weightDict={0:expectedCounts[0],1:expectedCounts[1]}
                weight0=np.full(len(mse),expectedCounts[0]/len(mse))
                weight1=np.full(len(mseCustom),expectedCounts[1]/len(mseCustom))
                
                h1=plt.hist(mse,weights=weight0,
                        bins=5000,range=[0,maxRange],
                        alpha=0.8,label='background',cumulative=-1)
                h2=plt.hist(mseCustom,weights=weight1,
                        bins=5000,range=[0,maxRange],
                        color='r',alpha=0.8,label='signal',cumulative=-1)
                plt.yscale('log')
                plt.ylabel('Cumulative event counts / 0.02')
                plt.xlabel('Classifier output')
                plt.legend()
                plt.savefig(os.path.join(self.output,'cumulativeWeightedDiscriminator.pdf'))
                plt.clf()

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
                systematic=0.1

                toPlot = Z(s,b,systematic)
                plt.plot((h1[1][:-1]+h1[1][1:])/2,toPlot)
                es = weight1[0]*np.sqrt(s/weight1[0])
                eb = weight0[0]*np.sqrt(b/weight0[0])
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
            

