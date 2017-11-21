import os
import pandas as pd
import math
from dfConvert import convertTree
from pandasPlotting.Plotter import Plotter
from pandasPlotting.dfFunctions import expandArrays
from linearAlgebraFunctions import gram
from root_numpy import rec2array

makeDfs=False
saveDfs=True #Save the dataframes if they're remade

makePlots=False

exceptions=[] #variables to miss out when plotting


if __name__=='__main__':

    #Either make the dataframes fresh from the trees or just read them in
    if makeDfs:
        print "Making DataFrames"

        signalFile = '/nfs/dust/cms/group/susy-desy/marco/training_sample_new/top_sample_0.root'
        bkgdFile = '/nfs/dust/cms/group/susy-desy/marco/training_sample_new/stop_sample_0.root'

        signal = convertTree(signalFile,signal=True,passFilePath=True,tlVectors = ['selJet','sel_lep'])
        bkgd = convertTree(bkgdFile,signal=False,passFilePath=True,tlVectors = ['selJet','sel_lep'])

        if saveDfs:
            print 'Saving the dataframes'
            # Save the dfs?
            if not os.path.exists('dfs'): os.makedirs('dfs')
            signal.to_pickle('dfs/signal.pkl')
            bkgd.to_pickle('dfs/bkgd.pkl')
    else:
        print "Loading DataFrames" 

        signal = pd.read_pickle('dfs/signal.pkl')
        bkgd = pd.read_pickle('dfs/signal.pkl')

    if makePlots:
        print "Making plots" 
        signalPlotter = Plotter(signal,'testPlots/signal',exceptions=exceptions)
        bkgdPlotter = Plotter(bkgd,'testPlots/bkgd',exceptions=exceptions)

        signalPlotter.plotAllHists1D(withErrors=True)
        signalPlotter.correlations()
        bkgdPlotter.plotAllHists1D(withErrors=True)
        bkgdPlotter.correlations()
        pass

    #Now add the relevant variables to the DFs (make gram matrix)

    #Make a matrix of J+L x J+L where J is the number of jets and L is the number of leptons
    #Store it as a numpy matrix in the dataframe 
    
    #Use function that takes (4x) arrays of objects (for E,px,py,pz) and returns matrix

    #Must store it as an array of arrays as 2D numpy objects can't be stored in pandas

    #print 'm',signal['selJet_m'][0]+signal['sel_lep_m'][0]
    signal['gram'] = signal.apply(lambda row: gram(row['sel_lep_e']+[row['MET']]+row['selJet_e'],\
        row['sel_lep_px']+[row['MET']*math.cos(row['METPhi'])]+row['selJet_px'],\
        row['sel_lep_py']+[row['MET']*math.sin(row['METPhi'])]+row['selJet_py'],\
        row['sel_lep_pz']+[0]+row['selJet_pz'],oneD=True),axis=1)

    bkgd['gram'] = bkgd.apply(lambda row: gram(row['sel_lep_e']+[row['MET']]+row['selJet_e'],\
        row['sel_lep_px']+[row['MET']*math.cos(row['METPhi'])]+row['selJet_px'],\
        row['sel_lep_py']+[row['MET']*math.sin(row['METPhi'])]+row['selJet_py'],\
        row['sel_lep_pz']+[0]+row['selJet_pz'],oneD=True),axis=1)

    #Now carry out machine learning (with some algo specific diagnostics)

    # Put the data in a format for the machine learning: 
    # combine signal and background with an extra column indicating which it is

    signal['signal'] = 1
    bkgd['signal'] = 0

    combined = pd.concat([signal,bkgd])

    #Choose the variables to train on

    #gram matrix
    combined = combined[['gram','signal']] 

    #Vanilla
    # combined=combined[['signal','HT','MET','METPhi','MT','MT2W','n_jet',
    #'n_bjet','sel_lep_pt','sel_lep_eta','sel_lep_phi',
    #'selJet_phi','selJet_pt','selJet_eta','selJet_m']]

    # flatten the arrays
    combined = expandArrays(combined) 
    
    #Now split pseudorandomly into training and testing

    #Start with a BDT from sklearn (ala TMVA)

    pass

    #and carry out a diagnostic of the results
    pass
