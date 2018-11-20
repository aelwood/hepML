import matplotlib
matplotlib.use('Agg')
import os
import pandas as pd
import numpy as np
import math
import time
from dfConvert import convertTree

from keras import callbacks

from pandasPlotting.Plotter import Plotter
from pandasPlotting.dfFunctions import expandArrays
from pandasPlotting.dtFunctions import featureImportance

from MlClasses.MlData import MlData
from MlClasses.Bdt import Bdt
from MlClasses.Dnn import Dnn
from MlClasses.ComparePerformances import ComparePerformances

from MlFunctions.DnnFunctions import significanceLoss,significanceLossInvert,significanceLoss2Invert,significanceLossInvertSqrt,significanceFull,asimovSignificanceLoss,asimovSignificanceLossInvert,asimovSignificanceFull,truePositive,falsePositive

from linearAlgebraFunctions import gram,addGramToFlatDF

#Callback to move onto the next batch after stopping
earlyStopping = callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=3)

limitSize=None#10000 #Make this an integer N_events if you want to limit input

systematic=0.2 #systematic for the asimov signficance

makePlots=False

#ML options
plotFeatureImportances=False
doBDT=False
doDNN=True
doCrossVal=False
makeLearningCurve=False
doGridSearch=False #if this is true do a grid search, if not use the configs

doRegression=False
regressionVars=['MT2W']#,'HT']

makeHistograms=False

normalLoss=True
sigLoss=False
sigLossInvert=False
sigLoss2Invert=False
asimovSigLoss=False
asimovSigLossInvert=False
asimovSigLossBothInvert=False
#asimovSigLossSysts=[0.01,0.05,0.1,0.2,0.3,0.4,0.5]
asimovSigLossSysts=[0.1,0.3,0.5]
variableBatchSigLossInvert=False

#If not doing the grid search
dnnConfigs={
    #'dnn':{'epochs':100,'batch_size':32,'dropOut':None,'l2Regularization':None,'hiddenLayers':[1.0]},
      # 'dnn_batch128':{'epochs':200,'batch_size':128,'dropOut':None,'l2Regularization':None,'hiddenLayers':[1.0]},
      #  'dnn_batch2048':{'epochs':200,'batch_size':2048,'dropOut':None,'l2Regularization':None,'hiddenLayers':[1.0]},
      'dnn_batch4096':{'epochs':200,'batch_size':4096,'dropOut':None,'l2Regularization':None,'hiddenLayers':[1.0]},
     #'dnn_batch1024':{'epochs':200,'batch_size':1024,'dropOut':None,'l2Regularization':None,'hiddenLayers':[1.0]},
    # 'dnn_batch8192':{'epochs':200,'batch_size':8192,'dropOut':None,'l2Regularization':None,'hiddenLayers':[1.0]},
    # 'dnn2l':{'epochs':40,'batch_size':32,'dropOut':None,'l2Regularization':None,'hiddenLayers':[1.0,1.0]},
    # 'dnn3l':{'epochs':40,'batch_size':32,'dropOut':None,'l2Regularization':None,'hiddenLayers':[1.0,1.0,1.0]},
    # 'dnn3l_batch1024':{'epochs':40,'batch_size':1024,'dropOut':None,'l2Regularization':None,'hiddenLayers':[1.0,1.0,1.0]},
    # 'dnn5l':{'epochs':40,'batch_size':32,'dropOut':None,'l2Regularization':None,'hiddenLayers':[1.0,1.0,1.0,1.0,1.0]},
    # 'dnn_2p0n':{'epochs':40,'batch_size':32,'dropOut':None,'l2Regularization':None,'hiddenLayers':[2.0]},
    # 'dnn2l_2p0n':{'epochs':50,'batch_size':32,'dropOut':None,'l2Regularization':None,'hiddenLayers':[2.0,2.0]},
     'dnn3l_2p0n':{'epochs':200,'batch_size':4096,'dropOut':None,'l2Regularization':None,'hiddenLayers':[2.0,2.0,2.0]},
    # 'dnn4l_2p0n':{'epochs':50,'batch_size':32,'dropOut':None,'l2Regularization':None,'hiddenLayers':[2.0,2.0,2.0,2.0]},
    # 'dnn5l_2p0n':{'epochs':50,'batch_size':32,'dropOut':None,'l2Regularization':None,'hiddenLayers':[2.0,2.0,2.0,2.0,2.0]},

    # 'dnn_l2Reg0p01':{'epochs':40,'batch_size':32,'dropOut':None,'l2Regularization':0.1,'hiddenLayers':[1.0]},
    # 'dnn2l_l2Reg0p01':{'epochs':40,'batch_size':32,'dropOut':None,'l2Regularization':0.1,'hiddenLayers':[1.0,1.0]},
    # 'dnn3l_l2Reg0p01':{'epochs':50,'batch_size':32,'dropOut':None,'l2Regularization':0.1,'hiddenLayers':[1.0,1.0,1.0]},
    # 'dnn5l_l2Reg0p01':{'epochs':50,'batch_size':32,'dropOut':None,'l2Regularization':0.1,'hiddenLayers':[1.0,1.0,1.0,1.0,1.0]},
    # 'dnn2l_2p0n_l2Reg0p01':{'epochs':40,'batch_size':32,'dropOut':None,'l2Regularization':0.1,'hiddenLayers':[2.0,2.0]},
    # 'dnn3l_2p0n_l2Reg0p01':{'epochs':50,'batch_size':32,'dropOut':None,'l2Regularization':0.1,'hiddenLayers':[2.0,2.0,2.0]},
    # 'dnn4l_2p0n_l2Reg0p01':{'epochs':50,'batch_size':32,'dropOut':None,'l2Regularization':0.1,'hiddenLayers':[2.0,2.0,2.0,2.0]},
    # 'dnn5l_2p0n_l2Reg0p01':{'epochs':50,'batch_size':32,'dropOut':None,'l2Regularization':0.1,'hiddenLayers':[2.0,2.0,2.0,2.0,2.0]},

    # 'dnndo0p5':{'epochs':10,'batch_size':32,'dropOut':0.5,'l2Regularization':None,'hiddenLayers':[1.0]},
    # 'dnn2ldo0p5':{'epochs':10,'batch_size':32,'dropOut':0.5,'l2Regularization':None,'hiddenLayers':[1.0,0.5]},
    # 'dnndo0p2':{'epochs':30,'batch_size':32,'dropOut':0.2,'l2Regularization':None,'hiddenLayers':[1.0]},
    # 'dnn2ldo0p2':{'epochs':30,'batch_size':32,'dropOut':0.2,'l2Regularization':None,'hiddenLayers':[1.0,1.0]},
    # 'dnn3ldo0p2':{'epochs':30,'batch_size':32,'dropOut':0.2,'l2Regularization':None,'hiddenLayers':[1.0,1.0,1.0]},
    # 'dnnSmall':{'epochs':20,'batch_size':32,'dropOut':None,'l2Regularization':None,'l2Regularization':None,'hiddenLayers':[0.3]},
    # 'dnn2lSmall':{'epochs':20,'batch_size':32,'dropOut':None,'l2Regularization':None,'hiddenLayers':[0.66,0.3]},
    # 'dnn3lSmall':{'epochs':40,'batch_size':32,'dropOut':None,'l2Regularization':None,'hiddenLayers':[0.66,0.5,0.3]},

    #Bests
    #4 vector
    # 'dnn3l_2p0n_do0p25':{'epochs':40,'batch_size':32,'dropOut':0.25,'l2Regularization':None,'hiddenLayers':[2.0,2.0,2.0]},
    # 'dnn3l_2p0n_do0p25_batch128':{'epochs':40,'batch_size':128,'dropOut':0.25,'l2Regularization':None,'hiddenLayers':[2.0,2.0,2.0]},
    # 'dnn3l_2p0n_do0p25_batch1024':{'epochs':40,'batch_size':1024,'dropOut':0.25,'l2Regularization':None,'hiddenLayers':[2.0,2.0,2.0]},
    # 'dnn3l_2p0n_do0p25_batch2048':{'epochs':40,'batch_size':2048,'dropOut':0.25,'l2Regularization':None,'hiddenLayers':[2.0,2.0,2.0]},
    #'dnn3l_2p0nIdo0p25_batch4096':{'epochs':200,'batch_size':4096,'dropOut':0.25,'l2Regularization':None,'hiddenLayers':[2.0,2.0,2.0]},
    #'dnn3l_2p0n_do0p25_batch8192':{'epochs':40,'batch_size':8192,'dropOut':0.25,'l2Regularization':None,'hiddenLayers':[2.0,2.0,2.0]},
    #'dnn5l_1p0n_do0p25':{'epochs':200,'batch_size':4096,'dropOut':0.25,'l2Regularization':None,'hiddenLayers':[1.0,1.0,1.0,1.0,1.0]},
    #'dnn4l_2p0n_do0p25':{'epochs':40,'batch_size':32,'dropOut':0.25,'l2Regularization':None,'hiddenLayers':[2.0,2.0,2.0,2.0]},
    #'dnn2lWide':{'epochs':30,'batch_size':32,'dropOut':0.25,'hiddenLayers':[2.0,2.0]},
        }

#If doing the grid search
def hiddenLayerGrid(nLayers,nNodes):
    hlg=[]
    for nn in nNodes:
        for nl in nLayers:
            hlg.append([nn for x in range(nl)])
        pass
    return hlg

dnnGridParams = dict(
        mlp__epochs=[10,20,50],
        mlp__batch_size=[32,64], 
        mlp__hiddenLayers=hiddenLayerGrid([1,2,3,4,5],[2.0,1.0,0.5]),
        mlp__dropOut=[None,0.25,0.5],
        # mlp__activation=['relu','sigmoid','tanh'],
        # mlp__optimizer=['adam','sgd','rmsprop'],
        ## NOT IMPLEMENTED YET:
        # mlp__learningRate=[0.5,1.0], 
        # mlp__weightConstraint=[1.0,3.0,5.0]
        )

bdtGridParams = dict(
        base_estimator__max_depth=[3,5],
        base_estimator__min_samples_leaf=[0.05,0.2],
        n_estimators=[400,800]
        )

if __name__=='__main__':

    #############################################################
    #Either make the dataframes fresh from the trees or just read them in
    print "Loading DataFrames" 

    combined=pd.read_csv('/nfs/dust/cms/user/elwoodad/higgsDataset/kaggle/atlas-higgs-challenge-2014-v2.csv')

    #Make 
    labelDict={'s':1,'b':0}
    combined['signal']=combined['Label'].map(labelDict)

    #Drop the useless labels
    combined=combined.drop(['Label','KaggleSet','KaggleWeight','EventId'],axis=1)

    #Change defaulted -999 values to NaN
    combined=combined.replace(-999.0,np.nan)

    #Define separate sets for convenience
    signal=combined[combined['signal']==1]
    bkgd=combined[combined['signal']==0]

    #Now sort out the weights
    #There should be two types of weights, the physical weights and the ML weights
    #The physical weights are the number of expected counts in the experiment
    #The ML weights take the relative weighting of the physical weights, but weight to the total number
    physicalWeights=combined['Weight']

    expectedSignal=combined[combined.signal==1]['Weight'].sum()
    expectedBkgd=combined[combined.signal==0]['Weight'].sum()

    nSignal=len(combined[combined.signal==1])
    nBkgd=len(combined[combined.signal==0])

    #Dictionary to store the ratio for the weights
    rDict={1:float(nSignal)/expectedSignal,0:float(nBkgd)/expectedBkgd}

    mlWeights=combined['Weight']*combined['signal'].map(rDict)

    combined=combined.drop('Weight',axis=1)

    if makePlots:
        print "Making plots" 
        #Skip out excessive jet info
        signalPlotter = Plotter(signal.copy(),'higgsPlots/signal')
        bkgdPlotter = Plotter(bkgd.copy(),'higgsPlots/bkgd')

        signalPlotter.plotAllHists1D()
        signalPlotter.correlations()
        bkgdPlotter.plotAllHists1D()
        bkgdPlotter.correlations()
        pass

    #############################################################
    #Carry out the organisation of the inputs or read them in if it's already done

    chosenVars = {
            'all':['signal','DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 
                'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 
                'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt', 'DER_pt_ratio_lep_tau', 
                'DER_met_phi_centrality', 'DER_lep_eta_centrality', 'PRI_tau_pt', 
                'PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 
                'PRI_met', 'PRI_met_phi', 'PRI_met_sumet', 'PRI_jet_num', 'PRI_jet_leading_pt', 
                'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt', 
                'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt'],

            }

    trainedModels={}

    for varSetName,varSet in chosenVars.iteritems():

        print ''
        print '==========================='
        print 'Analysing var set '+varSetName
        print '==========================='

        #Select just the features we're interested in
        #For now setting NaNs to 0 for compatibility
        combinedToRun = combined[varSet].copy()
        combinedToRun.fillna(0,inplace=True)

        # print columnsInDataFrame
        # print exit()

        #############################################################
        #Now everything is ready can start the machine learning

        if plotFeatureImportances:
            print 'Making feature importances'
            #Find the feature importance with a random forest classifier
            featureImportance(combinedToRun,'signal','higgsPlots/mlPlots/'+varSetName+'/featureImportance')

        print 'Splitting up data'

        #print mlWeights
        mlData = MlData(combinedToRun,'signal',weights=mlWeights.as_matrix())

        #Now split pseudorandomly into training and testing
        #Split the development set into training and testing
        #(forgetting about evaluation for now)

        mlData.prepare(evalSize=0.0,testSize=0.3,limitSize=limitSize)

        print 'Prepared data'

        if doBDT:

            if doGridSearch:
                print 'Running BDT grid search'
                bdt = Bdt(mlData,'higgsPlots/mlPlots/'+varSetName+'/bdtGridSearch')
                bdt.setup()
                bdt.gridSearch(param_grid=bdtGridParams,kfolds=3,n_jobs=4)

            elif not doRegression:
                #Start with a BDT from sklearn (ala TMVA)
                print 'Defining and fitting BDT'
                bdt = Bdt(mlData,'higgsPlots/mlPlots/'+varSetName+'/bdt')
                bdt.setup()
                bdt.fit()
                if doCrossVal:
                    print ' > Carrying out cross validation'
                    bdt.crossValidation(kfolds=5)
                if makeLearningCurve:
                    print ' > Making learning curves'
                    bdt.learningCurve(kfolds=5,n_jobs=3)

                #and carry out a diagnostic of the results
                print ' > Producing diagnostics'
                bdt.diagnostics()

                trainedModels[varSetName+'_bdt']=bdt

        if doDNN:

            if doGridSearch:
                print 'Running DNN grid search'
                dnn = Dnn(mlData,'higgsPlots/mlPlots/'+varSetName+'/dnnGridSearch')
                dnn.setup()
                dnn.gridSearch(param_grid=dnnGridParams,kfolds=3,epochs=20,batch_size=32,n_jobs=4)

            if doRegression:

                for name,config in dnnConfigs.iteritems():

                    for regressionVar in regressionVars:

                        if regressionVar not in varSet: continue

                        #Drop unconverged events for MT2
                        if regressionVar is 'MT2W': 
                            toRunRegression=combinedToRun[combinedToRun.MT2W!=999.0]
                        else:
                            toRunRegression=combinedToRun
                        

                        mlDataRegression = MlData(toRunRegression.drop('signal'),regressionVar)
                        mlDataRegression.prepare(evalSize=0.0,testSize=0.2,limitSize=limitSize,standardise=False)

                        print 'Defining and fitting DNN',name,'Regression',regressionVar
                        dnn = Dnn(mlDataRegression,'higgsPlots/mlPlots/regression/'+varSetName+'/'+name,doRegression=True)
                        dnn.setup(hiddenLayers=config['hiddenLayers'],dropOut=config['dropOut'],l2Regularization=config['l2Regularization'])
                        dnn.fit(epochs=config['epochs'],batch_size=config['batch_size'])
                        dnn.save()

                        if makeLearningCurve:
                            print ' > Making learning curves'
                            dnn.learningCurve(kfolds=3,n_jobs=1,scoring='neg_mean_squared_error')

                        print ' > Producing diagnostics'
                        dnn.diagnostics()

            
            else:
                #Now lets move on to a deep neural net 
                for name,config in dnnConfigs.iteritems():

                    if normalLoss:
                        print 'Defining and fitting DNN',name
                        dnn = Dnn(mlData,'higgsPlots/mlPlots/'+varSetName+'/'+name)
                        dnn.setup(hiddenLayers=config['hiddenLayers'],dropOut=config['dropOut'],l2Regularization=config['l2Regularization'],
                                extraMetrics=[
                                    significanceLoss(expectedSignal,expectedBkgd),significanceFull(expectedSignal,expectedBkgd),
                                    asimovSignificanceFull(expectedSignal,expectedBkgd,systematic),truePositive,falsePositive
                                    ])
                        dnn.fit(epochs=config['epochs'],batch_size=128,callbacks=[earlyStopping])
                        dnn.save()
                        if doCrossVal:
                            print ' > Carrying out cross validation'
                            dnn.crossValidation(kfolds=5,epochs=config['epochs'],batch_size=config['batch_size'])
                        if makeLearningCurve:
                            print ' > Making learning curves'
                            dnn.learningCurve(kfolds=5,n_jobs=1)

                        print ' > Producing diagnostics'
                        #dnn.explainPredictions()
                        dnn.diagnostics(batchSize=8192)
                        dnn.makeHepPlots(expectedSignal,expectedBkgd,asimovSigLossSysts,makeHistograms=makeHistograms)

                        trainedModels[varSetName+'_'+name]=dnn

                    if sigLoss:

                        print 'Defining and fitting DNN with significance loss function',name
                        dnn = Dnn(mlData,'higgsPlots/mlPlots/sigLoss/'+varSetName+'/'+name)
                        dnn.setup(hiddenLayers=config['hiddenLayers'],dropOut=config['dropOut'],l2Regularization=config['l2Regularization'],
                                loss=significanceLoss(expectedSignal,expectedBkgd),
                                extraMetrics=[
                                    significanceLoss(expectedSignal,expectedBkgd),significanceFull(expectedSignal,expectedBkgd),
                                    asimovSignificanceFull(expectedSignal,expectedBkgd,systematic),truePositive,falsePositive
                                ])
                        dnn.fit(epochs=config['epochs'],batch_size=config['batch_size'],callbacks=[earlyStopping])
                        dnn.save()
                        print ' > Producing diagnostics'
                        #dnn.explainPredictions()
                        dnn.diagnostics(batchSize=8192)
                        dnn.makeHepPlots(expectedSignal,expectedBkgd,asimovSigLossSysts,makeHistograms=makeHistograms)

                        trainedModels[varSetName+'_sigLoss_'+name]=dnn

                    if sigLossInvert:

                        print 'Defining and fitting DNN with significance loss function',name
                        dnn = Dnn(mlData,'higgsPlots/mlPlots/sigLossInvert/'+varSetName+'/'+name)
                        dnn.setup(hiddenLayers=config['hiddenLayers'],dropOut=config['dropOut'],l2Regularization=config['l2Regularization'],
                                loss=significanceLossInvert(expectedSignal,expectedBkgd),
                                extraMetrics=[
                                    significanceLoss(expectedSignal,expectedBkgd),significanceFull(expectedSignal,expectedBkgd),
                                    asimovSignificanceFull(expectedSignal,expectedBkgd,systematic),truePositive,falsePositive
                                ])
                        dnn.fit(epochs=config['epochs'],batch_size=config['batch_size'],callbacks=[earlyStopping])
                        dnn.save()
                        print ' > Producing diagnostics'
                        dnn.diagnostics(batchSize=8192)
                        dnn.makeHepPlots(expectedSignal,expectedBkgd,asimovSigLossSysts,makeHistograms=makeHistograms)

                        trainedModels[varSetName+'_sigLossInvert_'+name]=dnn

                    if sigLoss2Invert:

                        print 'Defining and fitting DNN with significance loss function',name
                        dnn = Dnn(mlData,'higgsPlots/mlPlots/sigLoss2Invert/'+varSetName+'/'+name)
                        dnn.setup(hiddenLayers=config['hiddenLayers'],dropOut=config['dropOut'],l2Regularization=config['l2Regularization'],
                                loss=significanceLoss2Invert(expectedSignal,expectedBkgd),
                                extraMetrics=[
                                    significanceLoss(expectedSignal,expectedBkgd),significanceFull(expectedSignal,expectedBkgd),
                                    asimovSignificanceFull(expectedSignal,expectedBkgd,systematic),truePositive,falsePositive
                                ])
                        dnn.fit(epochs=config['epochs'],batch_size=config['batch_size'],callbacks=[earlyStopping])
                        dnn.save()
                        print ' > Producing diagnostics'
                        dnn.diagnostics(batchSize=8192)
                        dnn.makeHepPlots(expectedSignal,expectedBkgd,asimovSigLossSysts,makeHistograms=makeHistograms)

                        trainedModels[varSetName+'_sigLoss2Invert_'+name]=dnn


                    if asimovSigLossInvert:

                        #First set up a model that trains on the sig loss
                        print 'Defining and fitting DNN with inverted asimov significance loss function',name
                        dnn = Dnn(mlData,'higgsPlots/mlPlots/asimovSigLossInvert/'+varSetName+'/'+name)
                        dnn.setup(hiddenLayers=config['hiddenLayers'],dropOut=config['dropOut'],l2Regularization=config['l2Regularization'],
                                loss=significanceLoss(expectedSignal,expectedBkgd),
                                extraMetrics=[
                                    asimovSignificanceLossInvert(expectedSignal,expectedBkgd,systematic),asimovSignificanceFull(expectedSignal,expectedBkgd,systematic),
                                    significanceFull(expectedSignal,expectedBkgd),truePositive,falsePositive
                                ])

                        dnn.fit(epochs=5,batch_size=config['batch_size'])
                        dnn.diagnostics(batchSize=8192,subDir='pretraining')
                        dnn.makeHepPlots(expectedSignal,expectedBkgd,systematic,makeHistograms=makeHistograms,subDir='pretraining')

                        #Now recompile the model with a different loss and train further
                        dnn.recompileModel(asimovSignificanceLossInvert(expectedSignal,expectedBkgd,systematic))
                        dnn.fit(epochs=config['epochs'],batch_size=config['batch_size'],callbacks=[earlyStopping])
                        dnn.save()
                        print ' > Producing diagnostics'
                        #dnn.explainPredictions()
                        dnn.diagnostics(batchSize=8192)
                        dnn.makeHepPlots(expectedSignal,expectedBkgd,[systematic],makeHistograms=makeHistograms)

                        trainedModels[varSetName+'_asimovSigLossInvert_'+name]=dnn

                    if asimovSigLossBothInvert:

                        for chosenSyst in asimovSigLossSysts:

                            systName = str(chosenSyst).replace('.','p')

                            #First set up a model that trains on the sig loss
                            print 'Defining and fitting DNN with inverted asimov significance loss function',name
                            dnn = Dnn(mlData,'higgsPlots/mlPlots/asimovSigLossBothInvertSyst'+systName+'/'+varSetName+'/'+name)
                            dnn.setup(hiddenLayers=config['hiddenLayers'],dropOut=config['dropOut'],l2Regularization=config['l2Regularization'],
                                    loss=significanceLoss2Invert(expectedSignal,expectedBkgd),
                                    extraMetrics=[
                                        asimovSignificanceLossInvert(expectedSignal,expectedBkgd,chosenSyst),asimovSignificanceFull(expectedSignal,expectedBkgd,chosenSyst),
                                        significanceFull(expectedSignal,expectedBkgd),truePositive,falsePositive
                                    ])

                            dnn.fit(epochs=5,batch_size=config['batch_size'])
                            dnn.diagnostics(batchSize=8192,subDir='pretraining')
                            dnn.makeHepPlots(expectedSignal,expectedBkgd,[chosenSyst],makeHistograms=makeHistograms,subDir='pretraining')

                            #Now recompile the model with a different loss and train further
                            dnn.recompileModel(asimovSignificanceLossInvert(expectedSignal,expectedBkgd,chosenSyst))
                            dnn.fit(epochs=config['epochs'],batch_size=config['batch_size'],callbacks=[earlyStopping])
                            dnn.save()
                            print ' > Producing diagnostics'
                            dnn.diagnostics(batchSize=8192)
                            dnn.makeHepPlots(expectedSignal,expectedBkgd,[chosenSyst],makeHistograms=makeHistograms)

                            trainedModels[varSetName+'_asimovSigLossBothInvert_'+name+systName]=dnn

                    if asimovSigLoss:

                        print 'Defining and fitting DNN with asimov significance loss function',name
                        dnn = Dnn(mlData,'higgsPlots/mlPlots/asimovSigLoss/'+varSetName+'/'+name)

                        #First set up a model that trains on the sig loss
                        dnn.setup(hiddenLayers=config['hiddenLayers'],dropOut=config['dropOut'],l2Regularization=config['l2Regularization'],
                                loss=significanceLoss(expectedSignal,expectedBkgd),
                                extraMetrics=[
                                    asimovSignificanceLoss(expectedSignal,expectedBkgd,systematic),asimovSignificanceFull(expectedSignal,expectedBkgd,systematic),
                                    significanceFull(expectedSignal,expectedBkgd),truePositive,falsePositive
                                ])

                        dnn.fit(epochs=5,batch_size=config['batch_size'])
                        dnn.diagnostics(batchSize=8192,subDir='pretraining')
                        dnn.makeHepPlots(expectedSignal,expectedBkgd,systematic,makeHistograms=makeHistograms,subDir='pretraining')

                        #Now recompile the model with a different loss and train further
                        dnn.recompileModel(asimovSignificanceLoss(expectedSignal,expectedBkgd,systematic))
                        dnn.fit(epochs=config['epochs'],batch_size=config['batch_size'])
                        dnn.save()
                        print ' > Producing diagnostics'
                        dnn.diagnostics(batchSize=8192)
                        dnn.makeHepPlots(expectedSignal,expectedBkgd,[systematic],makeHistograms=makeHistograms)

                        trainedModels[varSetName+'_asimovSigLoss_'+name]=dnn

                    if variableBatchSigLossInvert:

                        print 'Defining and fitting DNN with significance loss function',name
                        timingFile.write('\nTiming variable batch, with DNN '+name)
                        t0=time.time()

                        dnn = Dnn(mlData,'higgsPlots/mlPlots/variableBatchSigLossInvert/'+varSetName+'/'+name)

                        dnn.setup(hiddenLayers=config['hiddenLayers'],dropOut=config['dropOut'],l2Regularization=config['l2Regularization'],
                                loss=significanceLossInvert(expectedSignal,expectedBkgd),
                                extraMetrics=[
                                    significanceLoss(expectedSignal,expectedBkgd),significanceFull(expectedSignal,expectedBkgd),
                                    asimovSignificanceFull(expectedSignal,expectedBkgd,systematic),truePositive,falsePositive
                                ])


                        for batch_size in [128,512,1024,2048,4096,8192,16384]:
                            print '\nRunning with batch size '+str(batch_size)
                            dnn.fit(epochs=100,batch_size=batch_size,callbacks=[earlyStopping])
                            dnn.diagnostics(batchSize=4096,subDir='batchSize'+str(batch_size))

                        dnn.save()
                        print ' > Producing diagnostics'
                        dnn.diagnostics(batchSize=4096)
                        t1=time.time()
                        timingFile.write(': '+str(t1-t0)+' s\n')
                        dnn.makeHepPlots(expectedSignal,expectedBkgd,systematic,makeHistograms=makeHistograms)

                        trainedModels[varSetName+'_variableBatchSigLossInvert_'+name]=dnn
        

                pass

    pass # end of variable set loop

    #Compare all the results
    if not doGridSearch and not doRegression:

        # #Now compare all the different versions
        compareMl = ComparePerformances(trainedModels,output='higgsPlots/mlPlots/comparisons')

        compareMl.compareRoc(append='_all')
        compareMl.rankMethods()
        #

        compareMl.compareRoc(['gram_dnn','gram_dnn3l_2p0n_do0p25','gram_dnn5l_1p0n_do0p25','gram_dnn4l_2p0n_do0p25'],append='_gramOnly')
        compareMl.compareRoc(['fourVector_dnn','fourVector_dnn3l_2p0n_do0p25','fourVector_dnn5l_1p0n_do0p25','fourVector_dnn4l_2p0n_do0p25'],append='_fourVectorOnly')
        compareMl.compareRoc(['fourVector_dnn','fourVector_dnn3l_2p0n_do0p25','fourVector_dnn5l_1p0n_do0p25','fourVector_dnn4l_2p0n_do0p25'],append='_fourVectorOnly')


        # #compareMl.compareRoc(['gram_dnn2l','gramMT_dnn2l','gramHT_dnn2l','gramMT2W_dnn2l','gramBL_dnn2l'],append='_gramOnlyDNN2l')
        # compareMl.compareRoc(['gram_dnn2ldo0p2','gramMT_dnn2ldo0p2','gramHT_dnn2ldo0p2','gramMT2W_dnn2ldo0p2','gramBL_dnn2ldo0p2'],append='_gramOnlyDNN2ldo0p2')
        # compareMl.compareRoc(['gram_dnn3ldo0p2','gramMT_dnn3ldo0p2','gramHT_dnn3ldo0p2','gramMT2W_dnn3ldo0p2','gramBL_dnn3ldo0p2'],append='_gramOnlyDNN3ldo0p2')
        # compareMl.compareRoc(['gram_bdt','gramMT_bdt','gramHT_bdt','gramMT2W_bdt','gramBL_bdt'], append='_gramOnlyBDT')
        #
        # compareMl.compareRoc(['fourVector_dnn','fourVectorMT_dnn','fourVectorHT_dnn','fourVectorMT2W_dnn','fourVectorBL_dnn'],append='_fourVectorOnlyDNN')
        # #compareMl.compareRoc(['fourVector_dnn2l','fourVectorMT_dnn2l','fourVectorHT_dnn2l','fourVectorMT2W_dnn2l','fourVectorBL_dnn2l'],append='_fourVectorOnlyDNN2l')
        # compareMl.compareRoc(['fourVector_dnn2ldo0p2','fourVectorMT_dnn2ldo0p2','fourVectorHT_dnn2ldo0p2','fourVectorMT2W_dnn2ldo0p2','fourVectorBL_dnn2ldo0p2'],append='_fourVectorOnlyDNN2ldo0p2')
        # compareMl.compareRoc(['fourVector_dnn3ldo0p2','fourVectorMT_dnn3ldo0p2','fourVectorHT_dnn3ldo0p2','fourVectorMT2W_dnn3ldo0p2','fourVectorBL_dnn3ldo0p2'],append='_fourVectorOnlyDNN3ldo0p2')
        # compareMl.compareRoc(['fourVector_bdt','fourVectorMT_bdt','fourVectorHT_bdt','fourVectorMT2W_bdt','fourVectorBL_bdt'], append='_fourVectorOnlyBDT')
        #
        compareMl.compareRoc(['gram_dnn5l_1p0n_do0p25','gram_bdt',
            'fourVector_dnn3l_2p0n_do0p25','fourVector_bdt',
            'vanilla_dnn3l_2p0n_do0p25','vanilla_dnn5l_1p0n_do0p25','vanilla_bdt'],
            append='_vanillaComparisons')
        #

        #DNN study
        # compareMl = ComparePerformances(trainedModels,output='higgsPlots/mlPlots/dnnStudy')
        # compareMl.compareRoc(append='_all')
        # compareMl.rankMethods()
        #
        # #BDT study
        # compareMl = ComparePerformances(trainedModels,output='higgsPlots/mlPlots/bdtStudy')
        # compareMl.compareRoc(append='_all')
        # compareMl.rankMethods()

        pass
