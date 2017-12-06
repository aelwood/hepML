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

from linearAlgebraFunctions import gram,addGramToFlatDF
from root_numpy import rec2array


makeDfs=False
saveDfs=True #Save the dataframes if they're remade

makePlots=False

prepareInputs=False

#ML options
plotFeatureImportances=True
doBDT=True
doDNN=True

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

            'gramBL':['signal','gram','selJetB','lep_type'],

            #The 4 vectors only
            'fourVector':['signal',
            'sel_lep_pt','sel_lep_eta','sel_lep_phi','sel_lep_m',
            'selJet_phi','selJet_pt','selJet_eta','selJet_m','MET'],

            'fourVectorBL':['signal','lep_type','selJetB',
            'sel_lep_pt','sel_lep_eta','sel_lep_phi','sel_lep_m',
            'selJet_phi','selJet_pt','selJet_eta','selJet_m','MET'],

            #A vanilla analysis with HL variables and lead 3 jets
            'vanilla':['signal','HT','MET','MT','MT2W','n_jet','lep_type'
            'n_bjet','sel_lep_pt','sel_lep_eta','sel_lep_phi',
            'selJet_phi0','selJet_pt0','selJet_eta0','selJet_m0',
            'selJet_phi1','selJet_pt1','selJet_eta1','selJet_m1',
            'selJet_phi2','selJet_pt2','selJet_eta2','selJet_m2'],

            }

    for varSetName,varSet in chosenVars.iteritems():

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
        combinedTest = combined[columnsInDataFrame].copy()
        combinedTest.fillna(0,inplace=True)

        #############################################################
        #Now everything is ready can start the machine learning

        if plotFeatureImportances:
            print 'Making feature importances'
            #Find the feature importance with a random forest classifier
            featureImportance(combinedGram,'signal','testPlots/mlPlots/gram/featureImportance')
            featureImportance(combinedVanilla,'signal','testPlots/mlPlots/vanilla/featureImportance')

        print 'Splitting up data'
        #Choose the one to do 
        #gram
        # mlData = MlData(combinedGram,'signal')
        # outDir = 'gram'

        #Vanilla
        mlData = MlData(combinedVanilla,'signal')
        outDir = 'vanilla'

        #Now split pseudorandomly into training and testing
        #Split the development set into training and testing
        #(forgetting about evaluation for now)

        mlData.prepare(evalSize=0.0,testSize=0.33)

        if doBDT:

            #Start with a BDT from sklearn (ala TMVA)
            print 'Defining and fitting BDT'
            bdt = Bdt(mlData,'testPlots/mlPlots/'+outDir+'/bdt')
            bdt.setup()
            bdt.fit()

            #and carry out a diagnostic of the results
            print ' > Producing diagnostics'
            bdt.diagnostics()

        if doDNN:

            #Now lets move on to a deep neural net 
            print 'Defining and fitting DNN'
            dnn = Dnn(mlData,'testPlots/mlPlots/'+outDir+'/dnn')
            dnn.setup()
            dnn.fit()

            print ' > Producing diagnostics'
            dnn.diagnostics()

            pass

    pass # end of variable set loop
