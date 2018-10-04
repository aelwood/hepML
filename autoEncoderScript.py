import os

import numpy as np
import matplotlib
matplotlib.use('Agg') #this stops matplotlib trying to use Xwindows backend when running remotely
import matplotlib.pyplot as plt
import pandas as pd

from keras import callbacks

from MlClasses.MlData import MlData
from MlClasses.AutoEncoder import AutoEncoder

#===== Define some useful variables =====
makeAnglesRelative=True
#=====

output='exampleOut' # an output directory (then make it if it doesn't exist)
if not os.path.exists(output): os.makedirs(output)

#class to stop training of dnns early
earlyStopping = callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=5)

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
#dfFull = dfFull.sample(10000,random_state=42)

#Look at the variables in the trees:

print 'The keys are:'
print dfFull.keys()

if makeAnglesRelative:
    #Make everything relative to MET Phi
    for k in dfFull.keys():
        if 'phi' in k:
            plt.hist(dfFull[dfFull['signal']==0][k].fillna(0),bins=50,range=[-np.pi,np.pi],alpha=0.7,label='bkgd')
            plt.hist(dfFull[dfFull['signal']==1][k].fillna(0),bins=50,range=[-np.pi,np.pi],alpha=0.7,label='sig')
            plt.legend()
            plt.savefig('exampleOut/autoEncoder/'+k+'before.pdf')
            plt.clf()
            #print k, df[k].iloc[0:10],df['METPhi'].iloc[0:10],df[k].iloc[0:10]-df['METPhi'].iloc[0:10]

            dfFull[k]=np.mod(dfFull[k]-dfFull['METPhi'],2*np.pi)

            plt.hist(dfFull[dfFull['signal']==0][k].fillna(0),bins=50,range=[0,2*np.pi],alpha=0.7,label='bkgd')
            plt.hist(dfFull[dfFull['signal']==1][k].fillna(0),bins=50,range=[0,2*np.pi],alpha=0.7,label='sig')
            plt.legend()
            plt.savefig('exampleOut/autoEncoder/'+k+'.pdf')
            plt.clf()


dfBkgd = dfFull[dfFull['signal']==0]
dfSig = dfFull[dfFull['signal']==1]

#Define and select a subset of the variables:

subset=[#'HT','MET', #energy sums
        #'MT','MT2W', #topological variables
        'MET','METPhi',
        'n_jet','n_bjet', #jet and b-tag multiplicities
        'sel_lep_pt0','sel_lep_eta0','sel_lep_phi0', #lepton 4-vector
        'selJet_phi0','selJet_pt0','selJet_eta0','selJet_m0',# lead jet 4-vector
        'selJet_phi1','selJet_pt1','selJet_eta1','selJet_m1',# second jet 4-vector
        'selJet_phi2','selJet_pt2','selJet_eta2','selJet_m2',
        'selJet_phi3','selJet_pt3','selJet_eta3','selJet_m3',
        'selJet_phi4','selJet_pt4','selJet_eta4','selJet_m4',
        'selJet_phi5','selJet_pt5','selJet_eta5','selJet_m5',
        'selJet_phi6','selJet_pt6','selJet_eta6','selJet_m6',
        'selJet_phi7','selJet_pt7','selJet_eta7','selJet_m7',
        'selJet_phi8','selJet_pt8','selJet_eta8','selJet_m8',
        ]

df=dfBkgd[subset]

df=df.fillna(0)

print 'The reduced keys are:'
print df.keys()

#=============================================================
#===== Make a simple network to carry out classification =====
#=============================================================

print 'Running classification'

# here I make use of the hepML framework with keras
# aim is to correctly classify signal or background events

#===== Prepare the data =====
# use an MlData class to wrap and manipulate the data with easy functions

print 'Preparing data'

mlDataC = MlData(df,None) #insert the dataframe and tell it what the truth variable is
mlDataC.split(evalSize=0.0,testSize=0.3) #Split into train and test sets, leave out evaluation set for now

#Now decide whether we want to standardise the dataset
#it is worth seeing what happens to training with and without this option
#(this must be done after the split to avoid information leakage)
mlDataC.standardise()

#make input for autoencoder
mlDataC.makeAutoEncoder()

#===== Setup and run the network  =====

print 'Setting up network'

dnnC = AutoEncoder(mlDataC,os.path.join(output,'autoEncoder')) #give it the data and an output directory for plots

#build a 2 hidden layer model with 50 neurons each layer
#Note: if the number of neurons is a float it treats it as a proportion of the input
# loss is binary cross entropy and one sigmoid neuron is used for output
dnnC.setup(hiddenLayers=[50,20,5,20,50],dropOut=None,l2Regularization=None,loss='mean_squared_error') 

#fit the defined network with the data passed to it
#define an early stopping if the loss stops decreasing after 2 epochs
print 'Fitting'
dnnC.fit(epochs=100,batch_size=128,callbacks=[earlyStopping])


#now produce some diagnostics to see how it went
print 'Making diagnostics'
dnnC.diagnostics() #generic diagnostics
dnnC.plotError()

#Now prep and scale the signal data
dfSig = dfSig[subset]
dfSig = dfSig.sample(len(mlDataC.X_test),random_state=41)
dfSig=dfSig.fillna(0)
dfSig= mlDataC.scaler.transform(dfSig)
dnnC.plotError(customPred=True,customData=dfSig,expectedCounts=[expectedBkgd,expectedSignal])

## hep specific plots including sensitivity estimates with a flat systematic etc: 
#print '\nMaking HEP plots'
#dnnC.makeHepPlots(expectedSignal,expectedBkgd,systematics=[0.2],makeHistograms=False)


