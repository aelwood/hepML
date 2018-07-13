import os

import numpy as np
import matplotlib
matplotlib.use('Agg') #this stops matplotlib trying to use Xwindows backend when running remotely
import matplotlib.pyplot as plt
import pandas as pd

from keras import callbacks

from MlClasses.MlData import MlData
from MlClasses.Dnn import Dnn

#===== Define some useful variables =====

makePlots=True
doClassification=False
doRegression=True

output='exampleOut' # an output directory (then make it if it doesn't exist)
if not os.path.exists(output): os.makedirs(output)

#class to stop training of dnns early
earlyStopping = callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=2)

lumi=30. #luminosity in /fb
expectedBkgd=844000.*8.2e-4*lumi #cross section of ttbar sample in fb times efficiency measured by Marco

#ttbar background and stop (900,100) 
# df = pd.read_pickle('/nfs/dust/cms/user/elwoodad/dlNonCms/hepML/dfs/combined.pkl')
# expectedSignal=17.6*0.059*lumi #cross section of stop sample in fb times efficiency measured by Marco

#ttbar background and stop (600,400) 
dfFull = pd.read_pickle('/nfs/dust/cms/user/elwoodad/dlNonCms/hepML/dfs/combinedleonid.pkl')
expectedSignal=228.195*0.14*lumi 

#===== Load the data from a pickle file (choose one of the two below) =====

#Pick a subset of events to limit size for messing about
#be careful to pick randomly as the first half are signal and the second half background
dfFull = dfFull.sample(100000,random_state=42)

#Look at the variables in the trees:

print 'The keys are:'
print dfFull.keys()

#Define and select a subset of the variables:

subset=['signal', #1 for signal and 0 for background
        'HT','MET', #energy sums
        'MT','MT2W', #topological variables
        'n_jet','n_bjet', #jet and b-tag multiplicities
        'sel_lep_pt0','sel_lep_eta0','sel_lep_phi0', #lepton 4-vector
        'selJet_phi0','selJet_pt0','selJet_eta0','selJet_m0',# lead jet 4-vector
        'selJet_phi1','selJet_pt1','selJet_eta1','selJet_m1',# second jet 4-vector
        'selJet_phi2','selJet_pt2','selJet_eta2','selJet_m2']# third jet 4-vector

df=dfFull[subset]

print 'The reduced keys are:'
print df.keys()

if makePlots:
    #===== Make a couple of plots: =====

    #Calculate the weights for each event and add them to the dataframe
    signalWeight = expectedSignal/(df.signal==1).sum() #divide expected events by number in dataframe
    bkgdWeight   = expectedBkgd/(df.signal==0).sum()

    #Add a weights column with the correct weights for background and signal
    df['weight'] = df['signal']*signalWeight+(1-df['signal'])*bkgdWeight

    #Choose some variables to plot and loop over them
    varsToPlot = ['HT','MT','MET','sel_lep_pt0','selJet_pt0']

    for v in varsToPlot:

        print 'Plotting',v
        maxRange=max(df[v])
        #Plot the signal and background but stacked on top of each other
        plt.hist([df[df.signal==0][v],df[df.signal==1][v]], #Signal and background input
                label=['background','signal'],
                bins=50, range=[0.,maxRange], 
                stacked=True, color = ['g','r'],
                weights=[df[df.signal==0]['weight'],df[df.signal==1]['weight']]) #supply the weights
        plt.yscale('log')
        plt.xlabel(v)
        plt.legend()
        plt.savefig(os.path.join(output,'hist_'+v+'.pdf')) #save the histogram
        plt.clf() #Clear it for the next one

    df = df.drop('weight',axis=1) #drop the weight to stop inference from it as truth variable

if doClassification:

    #=============================================================
    #===== Make a simple network to carry out classification =====
    #=============================================================

    print 'Running classification'

    # here I make use of the hepML framework with keras
    # aim is to correctly classify signal or background events

    #===== Prepare the data =====
    # use an MlData class to wrap and manipulate the data with easy functions

    print 'Preparing data'

    mlDataC = MlData(df,'signal') #insert the dataframe and tell it what the truth variable is
    mlDataC.split(evalSize=0.0,testSize=0.3) #Split into train and test sets, leave out evaluation set for now

    #Now decide whether we want to standardise the dataset
    #it is worth seeing what happens to training with and without this option
    #(this must be done after the split to avoid information leakage)
    mlDataC.standardise()

    #===== Setup and run the network  =====

    print 'Setting up network'

    dnnC = Dnn(mlDataC,os.path.join(output,'classification')) #give it the data and an output directory for plots

    #build a 2 hidden layer model with 50 neurons each layer
    #Note: if the number of neurons is a float it treats it as a proportion of the input
    # loss is binary cross entropy and one sigmoid neuron is used for output
    dnnC.setup(hiddenLayers=[20,20],dropOut=None,l2Regularization=None,loss='binary_crossentropy') 

    #fit the defined network with the data passed to it
    #define an early stopping if the loss stops decreasing after 2 epochs
    print 'Fitting'
    dnnC.fit(epochs=100,batch_size=128,callbacks=[earlyStopping])

    #now produce some diagnostics to see how it went
    print 'Making diagnostics'
    dnnC.diagnostics() #generic diagnostics, ROC curves etc
    # hep specific plots including sensitivity estimates with a flat systematic etc: 
    print '\nMaking HEP plots'
    dnnC.makeHepPlots(expectedSignal,expectedBkgd,systematics=[0.2],makeHistograms=False)

if doRegression:

    #=========================================================
    #===== Make a simple network to carry out regression =====
    #=========================================================

    #now we've seen a classification example, try a similar thing with regression
    #try to predict a higher level variable from the low level inputs

    print 'Running regression'

    print 'Preparing data'

    #Just pick the 4-vectors to train on
    subset = ['HT']
    for k in dfFull.keys():
        for v in ['selJet','sel_lep']:
            if ' '+v in ' '+k: subset.append(k)

    print 'Using subset',subset
    df=dfFull[subset]

    df=df.fillna(0) #NaNs in the input cause problems

    #insert the dataframe without the background class and the variable for regression
    mlDataR = MlData(df,'HT') 

    mlDataR.split(evalSize=0.0,testSize=0.3) #Split into train and test sets, leave out evaluation set for now

    #Now decide whether we want to standardise the dataset
    #it is worth seeing what happens to training with and without this option
    #(this must be done after the split to avoid information leakage)
    #mlDataR.standardise() #find this causes problems with regression

    print 'Setting up network'

    dnnR=Dnn(mlDataR,os.path.join(output,'regression'),doRegression=True)

    #here sets up with mean squared error and a linear output neuron
    dnnR.setup(hiddenLayers=[20,20],dropOut=None,l2Regularization=None)#,loss='mean_squared_error')

    print 'Fitting'
    dnnR.fit(epochs=100,batch_size=128,callbacks = [earlyStopping])

    print 'Making diagnostics'
    dnnR.diagnostics() #make regression specific diagnostics

