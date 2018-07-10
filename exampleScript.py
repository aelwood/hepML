import os

import numpy as np
import matplotlib
matplotlib.use('Agg') #this stops matplotlib trying to use Xwindows backend when running remotely
import matplotlib.pyplot as plt
import pandas as pd

#===== Define some useful variables =====

output='exampleOut' # an output directory (then make it if it doesn't exist)
if not os.path.exists(output): os.makedirs(output)

lumi=30. #luminosity in /fb
expectedBkgd=844000.*8.2e-4*lumi #cross section of ttbar sample in fb times efficiency measured by Marco

#===== Load the data from a pickle file (choose one of the two below) =====

#ttbar background and stop (900,100) 
df = pd.read_pickle('/nfs/dust/cms/user/elwoodad/dlNonCms/hepML/dfs/combined.pkl')
expectedSignal=17.6*0.059*lumi #cross section of stop sample in fb times efficiency measured by Marco

#ttbar background and stop (600,400) 
#df = pd.read_pickle('/nfs/dust/cms/user/elwoodad/dlNonCms/hepML/dfs/combinedLeonid.pkl')
#expectedSignal=228.195*0.14*lumi 

#Pick a subset of events to limit size for messing about
#be careful to pick randomly as the first half are signal and the second half background
df = df.sample(10000,random_state=42)

#Look at the variables in the trees:

print 'The keys are:'
print df.keys()

#Define and select a subset of the variables:

subset=['signal', #1 for signal and 0 for background
        'HT','MET', #energy sums
        'MT','MT2W', #topological variables
        'n_jet','n_bjet', #jet and b-tag multiplicities
        'sel_lep_pt0','sel_lep_eta0','sel_lep_phi0', #lepton 4-vector
        'selJet_phi0','selJet_pt0','selJet_eta0','selJet_m0',# lead jet 4-vector
        'selJet_phi1','selJet_pt1','selJet_eta1','selJet_m1',# second jet 4-vector
        'selJet_phi2','selJet_pt2','selJet_eta2','selJet_m2']# third jet 4-vector

df=df[subset]

print 'The reduced keys are:'
print df.keys()

#===== Make a couple of plots: =====

#Calculate the weights for each event and add them to the dataframe
signalWeight = expectedSignal/(df.signal==1).sum() #divide expected events by number in dataframe
bkgdWeight   = expectedBkgd/(df.signal==0).sum()

#Add a weights column with the correct weights for background and signal
df['weight'] = df['signal']*signalWeight+(1-df['signal'])*bkgdWeight

#Choose some variables to plot and loop over them
varsToPlot = ['HT','MT']#,'MET','sel_lep_pt0','selJet_pt0']

for v in varsToPlot:

    print 'Plotting',v
    maxRange=max(df[v])
    #Plot the signal and background but stacked on top of each other
    plt.hist([df[df.signal==0][v],df[df.signal==1][v]], #Signal and background input
            label=['background','signal'],
            bins=50, range=[0.,maxRange], 
            stacked=True, color = ['g','r'],
            weights=[df[df.signal==1]['weight'],df[df.signal==0]['weight']]) #supply the weights
    plt.yscale('log')
    plt.xlabel(v)
    plt.legend()
    plt.savefig(os.path.join(output,'hist_'+v+'.pdf')) #save the histogram
    plt.clf() #Clear it for the next one

#
