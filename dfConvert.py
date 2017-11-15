#Convert a signal and background root tree to a dataframe
import os
import ROOT as r
import pandas as pd
from root_numpy import root2array, tree2array

#Function to convert a file path to a tree to a dataframe
def convertTree(treefile,output,signal=False):
    rfile = r.TFile(treefile) 
    intree = rfile.Get('outtree')
    df = pd.DataFrame(tree2array(intree))
    return df


#Run on its own for testing
if __name__=='__main__':
    signal = '/nfs/dust/cms/group/susy-desy/marco/training_sample_new/top_sample_0.root'
    background = '/nfs/dust/cms/group/susy-desy/marco/training_sample_new/stop_sample_0.root'
    if not os.path.exists('dfs'): os.makedirs('dfs')
    output='dfs'

    signalArray = convertTree(signal, output, signal=True)
    print signalArray

    bkgdArray = convertTree(background, output, signal=False)
    print bkgdArray

    #Produce some hists with pandas and root to check

