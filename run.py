import matplotlib
matplotlib.use('Agg')
import os
import pandas as pd
import numpy as np
import math
from dfConvert import convertTree

from pandasPlotting.Plotter import Plotter
from pandasPlotting.dfFunctions import expandArrays
from pandasPlotting.dtFunctions import featureImportance

from MlClasses.MlData import MlData
from MlClasses.Bdt import Bdt
from MlClasses.Dnn import Dnn
from MlClasses.ComparePerformances import ComparePerformances

from MlFunctions.DnnFunctions import significanceLoss

from linearAlgebraFunctions import gram,addGramToFlatDF
from root_numpy import rec2array


nInputFiles=20
limitSize=100000#None #Make this an integer N_events if you want to limit input

makeDfs=False
saveDfs=True #Save the dataframes if they're remade

makePlots=False

prepareInputs=False

#ML options
plotFeatureImportances=False
doBDT=False
doDNN=True
doCrossVal=False
makeLearningCurve=False
doGridSearch=False #if this is true do a grid search, if not use the configs

doRegression=False
regressionVars=['MT2W']#,'HT']

tryNewLoss=True

#If not doing the grid search
dnnConfigs={
    'dnn':{'epochs':40,'batch_size':32,'dropOut':None,'l2Regularization':None,'hiddenLayers':[1.0]},
    'dnn_batch128':{'epochs':40,'batch_size':128,'dropOut':None,'l2Regularization':None,'hiddenLayers':[1.0]},
    'dnn2l':{'epochs':40,'batch_size':32,'dropOut':None,'l2Regularization':None,'hiddenLayers':[1.0,1.0]},
    #'dnn3l':{'epochs':40,'batch_size':32,'dropOut':None,'l2Regularization':None,'hiddenLayers':[1.0,1.0,1.0]},
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
    'dnn3l_2p0n_do0p25':{'epochs':40,'batch_size':32,'dropOut':0.25,'l2Regularization':None,'hiddenLayers':[2.0,2.0,2.0]},
    'dnn3l_2p0n_do0p25_batch128':{'epochs':40,'batch_size':128,'dropOut':0.25,'l2Regularization':None,'hiddenLayers':[2.0,2.0,2.0]},
    'dnn5l_1p0n_do0p25':{'epochs':40,'batch_size':32,'dropOut':0.25,'l2Regularization':None,'hiddenLayers':[1.0,1.0,1.0,1.0,1.0]},
    'dnn4l_2p0n_do0p25':{'epochs':40,'batch_size':32,'dropOut':0.25,'l2Regularization':None,'hiddenLayers':[2.0,2.0,2.0,2.0]},
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
            signalFile.append('/nfs/dust/cms/group/susy-desy/marco/training_sample_new/stop_sample_'+str(i)+'.root')
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
            signal.to_pickle('dfs/signal.pkl')
            bkgd.to_pickle('dfs/bkgd.pkl')
    else:
        print "Loading DataFrames" 

        signal = pd.read_pickle('dfs/signal.pkl')
        bkgd = pd.read_pickle('dfs/bkgd.pkl')

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
            combined.to_pickle('dfs/combined.pkl')

    else:
        print 'Reading prepared files'
        combined = pd.read_pickle('dfs/combined.pkl')

    #Now carry out machine learning (with some algo specific diagnostics)
    #Choose the variables to train on

    chosenVars = {
            #Just the gram matrix, with or without b info
            'gram':['signal','gram'],

            # 'gramBL':['signal','gram','selJetB','lep_type'],
            #
            # 'gramMT':['signal','gram','MT'],
            #
            # 'gramMT2W':['signal','gram','MT2W'],
            #
            # 'gramHT':['signal','gram','HT'],
            #
            # #The 4 vectors only
            'fourVector':['signal',
            'sel_lep_pt','sel_lep_eta','sel_lep_phi','sel_lep_m',
            'selJet_phi','selJet_pt','selJet_eta','selJet_m','MET'],
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
            'vanilla':['signal','HT','MET','MT','MT2W','n_jet','lep_type'
            'n_bjet','sel_lep_pt','sel_lep_eta','sel_lep_phi',
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
                if ' '+v in ' '+k: columnsInDataFrame.append(k)

        #Select just the features we're interested in
        #For now setting NaNs to 0 for compatibility
        combinedToRun = combined[columnsInDataFrame].copy()
        combinedToRun.fillna(0,inplace=True)

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

        mlData.prepare(evalSize=0.2,testSize=0.2,limitSize=limitSize)

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

                        if makeLearningCurve:
                            print ' > Making learning curves'
                            dnn.learningCurve(kfolds=3,n_jobs=1,scoring='neg_mean_squared_error')

                        print ' > Producing diagnostics'
                        dnn.diagnostics()

            
            else:
                #Now lets move on to a deep neural net 
                for name,config in dnnConfigs.iteritems():
                    print 'Defining and fitting DNN',name
                    dnn = Dnn(mlData,'testPlots/mlPlots/'+varSetName+'/'+name)
                    dnn.setup(hiddenLayers=config['hiddenLayers'],dropOut=config['dropOut'],l2Regularization=config['l2Regularization'])
                    dnn.fit(epochs=config['epochs'],batch_size=config['batch_size'])
                    if doCrossVal:
                        print ' > Carrying out cross validation'
                        dnn.crossValidation(kfolds=5,epochs=config['epochs'],batch_size=config['batch_size'])
                    if makeLearningCurve:
                        print ' > Making learning curves'
                        dnn.learningCurve(kfolds=5,n_jobs=1)

                    print ' > Producing diagnostics'
                    dnn.diagnostics()


                    trainedModels[varSetName+'_'+name]=dnn

                    if tryNewLoss:

                        print 'Defining and fitting DNN with significance loss function'
                        dnn = Dnn(mlData,'testPlots/mlPlots/sigLoss/'+varSetName+'/'+name)
                        dnn.setup(hiddenLayers=config['hiddenLayers'],dropOut=config['dropOut'],l2Regularization=config['l2Regularization'],loss=significanceLoss)
                        dnn.fit(epochs=config['epochs'],batch_size=config['batch_size'])
                        print ' > Producing diagnostics'
                        dnn.diagnostics()
                        trainedModels[varSetName+'_sigLoss_'+name]=dnn

                pass

    pass # end of variable set loop

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
        # compareMl.compareRoc(['gram_dnn5l_1p0n_do0p25','gram_bdt',
        #     'fourVector_dnn3l_2p0n_do0p25','fourVector_bdt',
        #     'vanilla_dnn3l_2p0n_do0p25','vanilla_dnn5l_1p0n_do0p25','vanilla_bdt'],
        #     append='_vanillaComparisons')
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
