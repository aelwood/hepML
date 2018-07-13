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
from root_numpy import rec2array

timingFile = open('testPlots/timings.txt','w')
#Callback to move onto the next batch after stopping
earlyStopping = callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=2)

nInputFiles=100
limitSize=None#100000 #Make this an integer N_events if you want to limit input

#Use these to calculate the significance when it's used for training
#Taken from https://twiki.cern.ch/twiki/bin/view/CMS/SummerStudent2017#SUSY
# (dependent on batch size)
lumi=30. #luminosity in /fb
expectedSignal=17.6*0.059*lumi #cross section of stop sample in fb times efficiency measured by Marco
#expectedSignal=228.195*0.14*lumi #leonid's number
expectedBkgd=844000.*8.2e-4*lumi #cross section of ttbar sample in fb times efficiency measured by Marco
systematic=0.1 #systematic for the asimov signficance

makeDfs=False
saveDfs=True #Save the dataframes if they're remade
#appendInputName='leonid'
appendInputName=''

makePlots=False

prepareInputs=False
addGramMatrix=False

#ML options
plotFeatureImportances=False
doBDT=False
doDNN=True
doCrossVal=False
makeLearningCurve=False
doGridSearch=False #if this is true do a grid search, if not use the configs

doRegression=True
regressionVars=['MT2W']#,'HT']

makeHistograms=False

normalLoss=True
sigLoss=False
sigLossInvert=True
sigLoss2Invert=False
sigLossInvertSqrt=False
asimovSigLoss=False
asimovSigLossInvert=False
asimovSigLossBothInvert=True
asimovSigLossSysts=[0.01,0.05,0.1,0.2,0.3,0.4,0.5]
crossEntropyFirst=False
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
    # 'dnn3l_2p0n':{'epochs':50,'batch_size':32,'dropOut':None,'l2Regularization':None,'hiddenLayers':[2.0,2.0,2.0]},
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
    if makeDfs:
        print "Making DataFrames"

        signalFile = []#'/nfs/dust/cms/group/susy-desy/marco/training_sample_new/stop_sample_0.root'
        bkgdFile = []#'/nfs/dust/cms/group/susy-desy/marco/training_sample_new/top_sample_0.root'

        for i in range(nInputFiles):
            signalFile.append(' /nfs/dust/cms/group/susy-desy/marco/leonid/stop_600_400/stop_samples_'+str(i)+'.root')
            bkgdFile.append('/nfs/dust/cms/group/susy-desy/marco/training_sample_new/top_sample_'+str(i)+'.root')

        signal = convertTree(signalFile,signal=True,passFilePath=True,tlVectors = ['selJet','sel_lep'])
        bkgd = convertTree(bkgdFile,signal=False,passFilePath=True,tlVectors = ['selJet','sel_lep'])

        # #Expand the variables to 1D
        signal = expandArrays(signal) 
        bkgd = expandArrays(bkgd) 

        if saveDfs:
            print 'Saving the dataframes'
            # Save the dfs?
            if not os.path.exists('dfs'): os.makedirs('dfs')
            print 'signal size:',len(signal)
            signal.to_pickle('dfs/signal'+appendInputName+'.pkl')
            print 'bkgd size:',len(bkgd)
            bkgd.to_pickle('dfs/bkgd'+appendInputName+'.pkl')
    else:
        print "Loading DataFrames" 

        signal = pd.read_pickle('dfs/signal'+appendInputName+'.pkl')
        bkgd = pd.read_pickle('dfs/bkgd'+appendInputName+'.pkl')

    if makePlots:
        print "Making plots" 
        #Skip out excessive jet info
        exceptions=[] 
        for k in signal.keys():
            if 'selJet' in k or '_x' in k or '_y' in k or '_z' in k:
                exceptions.append(k)
        signalPlotter = Plotter(signal.copy(),'testPlots/signal',exceptions=exceptions)
        bkgdPlotter = Plotter(bkgd.copy(),'testPlots/bkgd',exceptions=exceptions)

        signalPlotter.plotAllHists1D(withErrors=True)
        signalPlotter.correlations()
        bkgdPlotter.plotAllHists1D(withErrors=True)
        bkgdPlotter.correlations()
        pass

    #############################################################
    #Carry out the organisation of the inputs or read them in if it's already done
    if prepareInputs:
        print 'Preparing inputs'

        # Put the data in a format for the machine learning: 
        # combine signal and background with an extra column indicating which it is

        signal['signal'] = 1
        bkgd['signal'] = 0

        combined = pd.concat([signal,bkgd])

        #Now add the relevant variables to the DFs (make gram matrix)

        #Make a matrix of J+L x J+L where J is the number of jets and L is the number of leptons
        #Store it as a numpy matrix in the dataframe 
        

        #METHOD 1:
        # Store as a matrix in the numpy array
        # It's better for it to be flat for machine learning... so using method 2

        #Use function that takes (4x) arrays of objects (for E,px,py,pz) and returns matrix
        #Must store it as an array of arrays as 2D numpy objects can't be stored in pandas
        
        #print 'm',signal['selJet_m'][0]+signal['sel_lep_m'][0]
        # signal['gram'] = signal.apply(lambda row: gram(row['sel_lep_e']+[row['MET']]+row['selJet_e'],\
        #     row['sel_lep_px']+[row['MET']*math.cos(row['METPhi'])]+row['selJet_px'],\
        #     row['sel_lep_py']+[row['MET']*math.sin(row['METPhi'])]+row['selJet_py'],\
        #     row['sel_lep_pz']+[0]+row['selJet_pz']),axis=1)
        #
        # bkgd['gram'] = bkgd.apply(lambda row: gram(row['sel_lep_e']+[row['MET']]+row['selJet_e'],\
        #     row['sel_lep_px']+[row['MET']*math.cos(row['METPhi'])]+row['selJet_px'],\
        #     row['sel_lep_py']+[row['MET']*math.sin(row['METPhi'])]+row['selJet_py'],\
        #     row['sel_lep_pz']+[0]+row['selJet_pz']),axis=1)

        #METHOD 2:
        #Put MET into the same format as the other objects
        if addGramMatrix:
            print 'Producing GRAM matrix'
            combined['MET_e']=combined['MET']
            combined.drop('MET',axis=1)#Drop the duplicate
            combined['MET_px']=combined['MET']*np.cos(combined['METPhi'])
            combined['MET_py']=combined['MET']*np.sin(combined['METPhi'])
            combined['MET_pz']=0
            nSelLep = 0
            nSelJet = 0
            for k in combined.keys():
                if 'sel_lep_px' in k: nSelLep+=1
                if 'selJet_px' in k: nSelJet+=1
            addGramToFlatDF(combined,single=['MET'],multi=[['sel_lep',nSelLep],['selJet',nSelJet]])

        if saveDfs:
            print 'Saving prepared files'
            combined.to_pickle('dfs/combined'+appendInputName+'.pkl')

    else:
        print 'Reading prepared files'
        combined = pd.read_pickle('dfs/combined'+appendInputName+'.pkl')

    #Now carry out machine learning (with some algo specific diagnostics)
    #Choose the variables to train on

    chosenVars = {
            #Just the gram matrix, with or without b info
            # 'gram':['signal','gram'],
            #
            # 'gramBL':['signal','gram','selJetB','lep_type'],
            #
            # 'gramMT':['signal','gram','MT'],
            #
            # 'gramMT2W':['signal','gram','MT2W'],
            #
            # 'gramHT':['signal','gram','HT'],
            #
            # #The 4 vectors only
            # 'fourVector':['signal',
            # 'sel_lep_pt','sel_lep_eta','sel_lep_phi','sel_lep_m',
            # 'selJet_phi','selJet_pt','selJet_eta','selJet_m','MET'],
            #
            # 'fourVectorBL':['signal','lep_type','selJetB',
            # 'sel_lep_pt','sel_lep_eta','sel_lep_phi','sel_lep_m',
            # 'selJet_phi','selJet_pt','selJet_eta','selJet_m','MET'],
            #
            # 'fourVectorMT':['signal',
            # 'sel_lep_pt','sel_lep_eta','sel_lep_phi','sel_lep_m',
            # 'selJet_phi','selJet_pt','selJet_eta','selJet_m','MET','MT'],
            #
            # 'fourVectorMT2W':['signal',
            # 'sel_lep_pt','sel_lep_eta','sel_lep_phi','sel_lep_m',
            # 'selJet_phi','selJet_pt','selJet_eta','selJet_m','MET','MT2W'],
            #
            # 'fourVectorHT':['signal',
            # 'sel_lep_pt','sel_lep_eta','sel_lep_phi','sel_lep_m',
            # 'selJet_phi','selJet_pt','selJet_eta','selJet_m','MET','HT'],
            #
            # #A vanilla analysis with HL variables and lead 3 jets
            'vanilla':['signal','HT','MET','MT','MT2W','n_jet',
            'n_bjet','sel_lep_pt0','sel_lep_eta0','sel_lep_phi0',
            'selJet_phi0','selJet_pt0','selJet_eta0','selJet_m0',
            'selJet_phi1','selJet_pt1','selJet_eta1','selJet_m1',
            'selJet_phi2','selJet_pt2','selJet_eta2','selJet_m2'],

            }

    trainedModels={}

    for varSetName,varSet in chosenVars.iteritems():

        print ''
        print '==========================='
        print 'Analysing var set '+varSetName
        print '==========================='

        #Pick out the expanded arrays
        columnsInDataFrame = []
        for k in combined.keys():
            for v in varSet:
                #Little trick to ensure only the start of the string is checked
                if varSetName is 'vanilla':
                    if ' '+v+' ' in ' '+k+' ': columnsInDataFrame.append(k)
                elif ' '+v in ' '+k: columnsInDataFrame.append(k)


        #Select just the features we're interested in
        #For now setting NaNs to 0 for compatibility
        combinedToRun = combined[columnsInDataFrame].copy()
        combinedToRun.fillna(0,inplace=True)

        # print columnsInDataFrame
        # print exit()

        #############################################################
        #Now everything is ready can start the machine learning

        if plotFeatureImportances:
            print 'Making feature importances'
            #Find the feature importance with a random forest classifier
            featureImportance(combinedToRun,'signal','testPlots/mlPlots/'+varSetName+'/featureImportance')

        print 'Splitting up data'

        mlData = MlData(combinedToRun,'signal')

        #Now split pseudorandomly into training and testing
        #Split the development set into training and testing
        #(forgetting about evaluation for now)

        mlData.prepare(evalSize=0.0,testSize=0.3,limitSize=limitSize)

        if doBDT:

            if doGridSearch:
                print 'Running BDT grid search'
                bdt = Bdt(mlData,'testPlots/mlPlots/'+varSetName+'/bdtGridSearch')
                bdt.setup()
                bdt.gridSearch(param_grid=bdtGridParams,kfolds=3,n_jobs=4)

            elif not doRegression:
                #Start with a BDT from sklearn (ala TMVA)
                print 'Defining and fitting BDT'
                bdt = Bdt(mlData,'testPlots/mlPlots/'+varSetName+'/bdt')
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
                dnn = Dnn(mlData,'testPlots/mlPlots/'+varSetName+'/dnnGridSearch')
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
                        dnn = Dnn(mlDataRegression,'testPlots/mlPlots/regression/'+varSetName+'/'+name,doRegression=True)
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
                        dnn = Dnn(mlData,'testPlots/mlPlots/'+varSetName+'/'+name)
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
                        dnn.explainPredictions()
                        dnn.diagnostics(batchSize=8192)
                        dnn.makeHepPlots(expectedSignal,expectedBkgd,asimovSigLossSysts,makeHistograms=makeHistograms)

                        trainedModels[varSetName+'_'+name]=dnn

                    if sigLoss:

                        print 'Defining and fitting DNN with significance loss function',name
                        timingFile.write('\nTiming normal batch, with DNN '+name)
                        t0=time.time()
                        dnn = Dnn(mlData,'testPlots/mlPlots/sigLoss/'+varSetName+'/'+name)
                        dnn.setup(hiddenLayers=config['hiddenLayers'],dropOut=config['dropOut'],l2Regularization=config['l2Regularization'],
                                loss=significanceLoss(expectedSignal,expectedBkgd),
                                extraMetrics=[
                                    significanceLoss(expectedSignal,expectedBkgd),significanceFull(expectedSignal,expectedBkgd),
                                    asimovSignificanceFull(expectedSignal,expectedBkgd,systematic),truePositive,falsePositive
                                ])
                        dnn.fit(epochs=config['epochs'],batch_size=config['batch_size'],callbacks=[earlyStopping])
                        dnn.save()
                        print ' > Producing diagnostics'
                        dnn.explainPredictions()
                        dnn.diagnostics(batchSize=8192)
                        t1=time.time()
                        timingFile.write(': '+str(t1-t0)+' s\n')
                        dnn.makeHepPlots(expectedSignal,expectedBkgd,asimovSigLossSysts,makeHistograms=makeHistograms)

                        trainedModels[varSetName+'_sigLoss_'+name]=dnn

                    if sigLossInvert:

                        print 'Defining and fitting DNN with significance loss function',name
                        dnn = Dnn(mlData,'testPlots/mlPlots/sigLossInvert/'+varSetName+'/'+name)
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
                        dnn = Dnn(mlData,'testPlots/mlPlots/sigLoss2Invert/'+varSetName+'/'+name)
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


                    if sigLossInvertSqrt:

                        print 'Defining and fitting DNN with significance loss function',name
                        dnn = Dnn(mlData,'testPlots/mlPlots/sigLossInvertSqrt/'+varSetName+'/'+name)
                        dnn.setup(hiddenLayers=config['hiddenLayers'],dropOut=config['dropOut'],l2Regularization=config['l2Regularization'],
                                loss=significanceLossInvertSqrt(expectedSignal,expectedBkgd),
                                extraMetrics=[
                                    significanceLoss(expectedSignal,expectedBkgd),significanceFull(expectedSignal,expectedBkgd),
                                    asimovSignificanceFull(expectedSignal,expectedBkgd,systematic),truePositive,falsePositive
                                ])
                        dnn.fit(epochs=config['epochs'],batch_size=config['batch_size'])
                        dnn.save()
                        print ' > Producing diagnostics'
                        dnn.diagnostics(batchSize=8192)
                        dnn.makeHepPlots(expectedSignal,expectedBkgd,asimovSigLossSysts,makeHistograms=makeHistograms)

                        trainedModels[varSetName+'_sigLossInvertSqrt_'+name]=dnn
                    

                    if asimovSigLossInvert:

                        #First set up a model that trains on the sig loss
                        print 'Defining and fitting DNN with inverted asimov significance loss function',name
                        dnn = Dnn(mlData,'testPlots/mlPlots/asimovSigLossInvert/'+varSetName+'/'+name)
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
                        dnn.explainPredictions()
                        dnn.diagnostics(batchSize=8192)
                        dnn.makeHepPlots(expectedSignal,expectedBkgd,[systematic],makeHistograms=makeHistograms)

                        trainedModels[varSetName+'_asimovSigLossInvert_'+name]=dnn

                    if asimovSigLossBothInvert:

                        for chosenSyst in asimovSigLossSysts:

                            systName = str(chosenSyst).replace('.','p')

                            #First set up a model that trains on the sig loss
                            print 'Defining and fitting DNN with inverted asimov significance loss function',name
                            dnn = Dnn(mlData,'testPlots/mlPlots/asimovSigLossBothInvertSyst'+systName+'/'+varSetName+'/'+name)
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
                        dnn = Dnn(mlData,'testPlots/mlPlots/asimovSigLoss/'+varSetName+'/'+name)

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

                    if crossEntropyFirst:

                        #First train and fit with the cross entropy loss function
                        print 'Defining and fitting DNN',name
                        dnn = Dnn(mlData,'testPlots/mlPlots/crossEntropyFirst/'+varSetName+'/'+name)
                        dnn.setup(hiddenLayers=config['hiddenLayers'],dropOut=config['dropOut'],l2Regularization=config['l2Regularization'],
                                extraMetrics=[
                                    significanceLoss(expectedSignal,expectedBkgd),significanceFull(expectedSignal,expectedBkgd),
                                    asimovSignificanceFull(expectedSignal,expectedBkgd,systematic),truePositive,falsePositive
                                    ])
                        dnn.fit(epochs=config['epochs'],batch_size=128)
                        print ' > Producing diagnostics'
                        dnn.diagnostics(batchSize=128,subDir='pretraining')
                        dnn.makeHepPlots(expectedSignal,expectedBkgd,systematic,makeHistograms=makeHistograms,subDir='pretraining')

                        #Now cut away the obvious background, weight up the background events and retrain

                        #Remove all x entropy background
                        dataToPredict = combinedToRun.drop('signal',axis=1)
                        dataToPredict = dataToPredict.loc[:,~dataToPredict.columns.duplicated()]
                        # print '======'
                        # print dataToPredict.columns
                        # print dnn.data.X_train.columns
                        dataToPredict = dnn.data.scaler.transform(dataToPredict.as_matrix())
                        toRunXEntropyFirst = combinedToRun[dnn.model.predict(dataToPredict)>0.5]

                        #weight up the background based on the inbalance
                        nSignal = len(toRunXEntropyFirst[toRunXEntropyFirst.signal==1])
                        nBkgd = len(toRunXEntropyFirst[toRunXEntropyFirst.signal==0])
                        upWeight = float(nSignal)/nBkgd

                        def fabricateWeight(signal,weight):
                            if signal==0: return weight
                            else: return 1.0

                        weights = toRunXEntropyFirst.apply(lambda row: fabricateWeight(row.signal,upWeight),axis=1) 

                        mlDataXEntropyFirst = MlData(toRunXEntropyFirst,'signal',weights=weights.as_matrix())
                        mlDataXEntropyFirst.prepare(evalSize=0.0,testSize=0.3,limitSize=limitSize)

                        print 'Defining and fitting DNN with significance loss function',name
                        dnn2 = Dnn(mlDataXEntropyFirst,'testPlots/mlPlots/crossEntropyFirst/'+varSetName+'/'+name)
                        dnn2.setup(hiddenLayers=config['hiddenLayers'],dropOut=config['dropOut'],l2Regularization=config['l2Regularization'],
                                loss=significanceLoss(expectedSignal,expectedBkgd),
                                extraMetrics=[
                                    significanceLoss(expectedSignal,expectedBkgd),significanceFull(expectedSignal,expectedBkgd),
                                    asimovSignificanceFull(expectedSignal,expectedBkgd,systematic),truePositive,falsePositive
                                ])
                        dnn2.fit(epochs=config['epochs'],batch_size=config['batch_size'])
                        print ' > Producing diagnostics'
                        dnn2.diagnostics(batchSize=8192)

                        #Now need to make the HEP plots with the full dataset by passing a custom prediction that includes the decision from dnn2
                        # In this case we take xEntropy Score > 0.5 and then the sig loss output
                        # i.e. if xEntropyScore < 0.5 return 0, else return sig loss output

                        #save the prediction from dnn
                        #firstPred = dnn.testPrediction()
                        firstPred = dnn.model.predict_classes(dnn.data.X_test.as_matrix())
                        #unscale the data in dnn
                        dataForPred2=dnn.data.scaler.inverse_transform(dnn.data.X_test)
                        #rescale the data to dnn2
                        dataForPred2=dnn2.data.scaler.transform(dataForPred2)
                        #predict dnn2
                        secondPred = dnn2.model.predict(dataForPred2)
                        #combine the predictions
                        finalPred = firstPred*secondPred

                        dnn.makeHepPlots(expectedSignal,expectedBkgd,systematic,makeHistograms=makeHistograms,customPrediction=finalPred)

                    if variableBatchSigLossInvert:

                        print 'Defining and fitting DNN with significance loss function',name
                        timingFile.write('\nTiming variable batch, with DNN '+name)
                        t0=time.time()

                        dnn = Dnn(mlData,'testPlots/mlPlots/variableBatchSigLossInvert/'+varSetName+'/'+name)

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
        compareMl = ComparePerformances(trainedModels,output='testPlots/mlPlots/comparisons')

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
        # compareMl = ComparePerformances(trainedModels,output='testPlots/mlPlots/dnnStudy')
        # compareMl.compareRoc(append='_all')
        # compareMl.rankMethods()
        #
        # #BDT study
        # compareMl = ComparePerformances(trainedModels,output='testPlots/mlPlots/bdtStudy')
        # compareMl.compareRoc(append='_all')
        # compareMl.rankMethods()

        pass
