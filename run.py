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

from linearAlgebraFunctions import gram,addGramToFlatDF
from root_numpy import rec2array


makeDfs=False
saveDfs=True #Save the dataframes if they're remade

makePlots=False

prepareInputs=False

#ML options
plotFeatureImportances=False
doBDT=True
doDNN=True

dnnConfigs={
    'dnn':{'epochs':50,'batch_size':32,'dropOut':None,'hiddenLayers':[1.0]},
    # 'dnn2l':{'epochs':50,'batch_size':32,'dropOut':None,'hiddenLayers':[0.66,0.66]},
    # 'dnndo':{'epochs':100,'batch_size':32,'dropOut':0.5,'hiddenLayers':[1.0]},
    # 'dnn2ldo':{'epochs':100,'batch_size':32,'dropOut':0.5,'hiddenLayers':[0.66,0.66]},
        }

if __name__=='__main__':

    #############################################################
    #Either make the dataframes fresh from the trees or just read them in
    if makeDfs:
        print "Making DataFrames"

        signalFile = '/nfs/dust/cms/group/susy-desy/marco/training_sample_new/top_sample_0.root'
        bkgdFile = '/nfs/dust/cms/group/susy-desy/marco/training_sample_new/stop_sample_0.root'

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
            # 'vanilla':['signal','HT','MET','MT','MT2W','n_jet','lep_type'
            # 'n_bjet','sel_lep_pt','sel_lep_eta','sel_lep_phi',
            # 'selJet_phi0','selJet_pt0','selJet_eta0','selJet_m0',
            # 'selJet_phi1','selJet_pt1','selJet_eta1','selJet_m1',
            # 'selJet_phi2','selJet_pt2','selJet_eta2','selJet_m2'],

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

        mlData.prepare(evalSize=0.0,testSize=0.33)

        #As the splitting is done with the same pseudorandom initialisation the test set is always the same
        if not 'truth' in trainedModels: trainedModels['truth']=mlData.y_test

        if doBDT:

            #Start with a BDT from sklearn (ala TMVA)
            print 'Defining and fitting BDT'
            bdt = Bdt(mlData,'testPlots/mlPlots/'+varSetName+'/bdt')
            bdt.setup()
            bdt.fit()

            #and carry out a diagnostic of the results
            print ' > Producing diagnostics'
            bdt.diagnostics()

            trainedModels[varSetName+'_bdt']=bdt.testPrediction()

        if doDNN:

            #Now lets move on to a deep neural net 
            
            # #Start with a 1 layer
            # print 'Defining and fitting DNN'
            # dnn = Dnn(mlData,'testPlots/mlPlots/'+varSetName+'/dnn')
            # dnn.setup()
            # dnn.fit(epochs=50,batch_size=32)
            #
            # print ' > Producing diagnostics'
            # dnn.diagnostics()
            #
            # trainedModels[varSetName+'_dnn']=dnn.testPrediction()
            #
            # #Now do 2 layer
            # print ''
            # print 'Defining and fitting DNN with 2 layers'
            # dnn2l = Dnn(mlData,'testPlots/mlPlots/'+varSetName+'/dnn2l')
            # dnn2l.setup(hiddenLayers=[0.66,0.66])
            # dnn2l.fit(epochs=50,batch_size=32)
            #
            # print ' > Producing diagnostics'
            # dnn2l.diagnostics()
            #
            # trainedModels[varSetName+'_dnn2l']=dnn2l.testPrediction()
            #

            for name,config in dnnConfigs.iteritems():
                print 'Defining and fitting DNN',name
                dnn = Dnn(mlData,'testPlots/mlPlots/'+varSetName+'/'+name)
                dnn.setup(hiddenLayers=config['hiddenLayers'],dropOut=config['dropOut'])
                dnn.fit(epochs=config['epochs'],batch_size=config['batch_size'])

                print ' > Producing diagnostics'
                dnn.diagnostics()

                trainedModels[varSetName+'_'+name]=dnn.testPrediction()

            pass

    pass # end of variable set loop

    #Now compare all the different versions
    # compareMl = ComparePerformances(trainedModels,output='testPlots/mlPlots/comparisons_tuning')
    #
    # compareMl.compareRoc(append='_all')
    #
    # compareMl.compareRoc(['gram_dnn','gramMT_dnn','gramHT_dnn','gramMT2W_dnn','gramBL_dnn'],append='_gramOnlyDNN')
    # compareMl.compareRoc(['gram_dnn2l','gramMT_dnn2l','gramHT_dnn2l','gramMT2W_dnn2l','gramBL_dnn2l'],append='_gramOnlyDNN2l')
    # compareMl.compareRoc(['gram_bdt','gramMT_bdt','gramHT_bdt','gramMT2W_bdt','gramBL_bdt'], append='_gramOnlyBDT')
    #
    # compareMl.compareRoc(['fourVector_dnn','fourVectorMT_dnn','fourVectorHT_dnn','fourVectorMT2W_dnn','fourVectorBL_dnn'],append='_fourVectorOnlyDNN')
    # compareMl.compareRoc(['fourVector_dnn2l','fourVectorMT_dnn2l','fourVectorHT_dnn2l','fourVectorMT2W_dnn2l','fourVectorBL_dnn2l'],append='_fourVectorOnlyDNN2l')
    # compareMl.compareRoc(['fourVector_bdt','fourVectorMT_bdt','fourVectorHT_bdt','fourVectorMT2W_bdt','fourVectorBL_bdt'], append='_fourVectorOnlyBDT')
    #
    # compareMl.compareRoc(['gram_dnn','gram_dnn2l','gram_bdt',
    #     'fourVector_dnn','fourVector_dnn2l','fourVector_bdt',
    #     'vanilla_dnn','vanilla_dnn2l','vanilla_bdt'],append='_vanillaComparisons')
    #

    #DNN study
    compareMl = ComparePerformances(trainedModels,output='testPlots/mlPlots/dnnStudy')
    compareMl.compareRoc(append='_all')

