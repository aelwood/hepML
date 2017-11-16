#Convert a signal and background root tree to a dataframe
import os
import ROOT as r
import pandas as pd
import matplotlib.pyplot as plt
from root_numpy import root2array, tree2array

#Function to convert a file path to a tree to a dataframe
def convertTree(tree,signal=False,passFilePath=False):
    if passFilePath:
        rfile = r.TFile(tree) 
        tree = rfile.Get('outtree')
    df = pd.DataFrame(tree2array(tree))
    return df


#Run on its own for testing
if __name__=='__main__':
    signal = '/nfs/dust/cms/group/susy-desy/marco/training_sample_new/top_sample_0.root'
    backgroundFile = '/nfs/dust/cms/group/susy-desy/marco/training_sample_new/stop_sample_0.root'

    signalArray = convertTree(signal,  signal=True,passFilePath=True)
    print signalArray

    backgroundFile = r.TFile(backgroundFile)
    background = backgroundFile.Get('outtree')
    bkgdArray = convertTree(background,  signal=False,passFilePath=False)
    print bkgdArray

    # Save the dfs?
    # if not os.path.exists('dfs'): os.makedirs('dfs')
    # output='dfs'

    #Produce some hists with pandas and root to check
    if not os.path.exists('testPlots'): os.makedirs('testPlots')

    c = r.TCanvas()
    background.Draw('HT>>(100,0,3500)')
    c.SaveAs('testPlots/rootHT.png')

    bkgdArray.hist('HT',bins=range(0,3500+35,35))
    plt.show()
    plt.savefig('testPlots/pandasHT.png')


