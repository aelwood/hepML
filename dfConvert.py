#Convert a signal and background root tree to a dataframe
import os
import ROOT as r
from array import array
import pandas as pd
import matplotlib.pyplot as plt
from root_numpy import root2array, tree2array

#Add the important components of the tlorentzvectors to the dataframe
def addTLorentzVectors(df,tree,branches):
    nentries=tree.GetEntries()
    arrDict = {}
    #Initialise empty arrays
    for b in branches:
        arrDict[b+'_pt'] = []
        arrDict[b+'_px'] = []
        arrDict[b+'_py'] = []
        arrDict[b+'_pz'] = []
        arrDict[b+'_phi'] = []
        arrDict[b+'_eta'] = []
        arrDict[b+'_m'] = []
        arrDict[b+'_e'] = []

    #Loop over the tree and fill the arrays with the info from the vectors
    for i,event in enumerate(tree):
        for b in branches:
            arrDict[b+'_pt'].append( [o.Pt() for o in getattr(event,b)])
            arrDict[b+'_px'].append( [o.Px() for o in getattr(event,b)])
            arrDict[b+'_py'].append( [o.Py() for o in getattr(event,b)])
            arrDict[b+'_pz'].append( [o.Pz() for o in getattr(event,b)])
            arrDict[b+'_phi'].append( [o.Phi() for o in getattr(event,b)])
            arrDict[b+'_eta'].append( [o.Eta() for o in getattr(event,b)])
            arrDict[b+'_m'].append( [o.M() for o in getattr(event,b)])
            arrDict[b+'_e'].append( [o.E() for o in getattr(event,b)])
        pass
    pass

    #Add the info to the data frames
    for b in branches:
        df[b+'_pt'] = arrDict[b+'_pt'] 
        df[b+'_px'] = arrDict[b+'_px'] 
        df[b+'_py'] = arrDict[b+'_py'] 
        df[b+'_pz'] = arrDict[b+'_pz'] 
        df[b+'_phi'] = arrDict[b+'_phi']
        df[b+'_eta'] = arrDict[b+'_eta']
        df[b+'_m'] = arrDict[b+'_m']
        df[b+'_e'] = arrDict[b+'_e']
             
    pass

#Function to convert a file path to a tree to a dataframe
def convertTree(tree,signal=False,passFilePath=False,tlVectors=[]):
    if passFilePath:
        if isinstance(tree,list):
            chain = r.TChain('outtree')
            for t in tree:
                chain.Add(t)
            tree=chain
        else:
            rfile = r.TFile(tree) 
            tree = rfile.Get('outtree')
    #Note that this step can be replaced by root_pandas
    # this can also flatten the trees automatically
    df = pd.DataFrame(tree2array(tree))
    if len(tlVectors)>0: addTLorentzVectors(df,tree,branches=tlVectors)
    return df
    

#Run on its own for testing
if __name__=='__main__':
    signal = ['/nfs/dust/cms/group/susy-desy/marco/training_sample_new/top_sample_0.root',
        '/nfs/dust/cms/group/susy-desy/marco/training_sample_new/top_sample_1.root']
    backgroundFile = ['/nfs/dust/cms/group/susy-desy/marco/training_sample_new/stop_sample_0.root',
        '/nfs/dust/cms/group/susy-desy/marco/training_sample_new/stop_sample_1.root',
            ]

    signalArray = convertTree(signal,  signal=True,passFilePath=True)
    print signalArray

    bkgdArray = convertTree(backgroundFile,  signal=False,passFilePath=True,tlVectors = ['selJet','sel_lep'])
    print bkgdArray

    print bkgdArray['selJet_pt']
    print bkgdArray['selJet_pt'][0][0]

    #Produce some hists with pandas and root to check
    if not os.path.exists('testPlots'): os.makedirs('testPlots')

    # print 'Checking root and pandas make the same histograms'
    # c = r.TCanvas()
    # background.Draw('HT>>(100,0,3500)')
    # c.SaveAs('testPlots/rootHT.png')
    #
    # bkgdArray.hist('HT',bins=range(0,3500+35,35))
    # plt.show()
    # plt.savefig('testPlots/pandasHT.png')
    #
    
    print 'Checking the sum of the jets from the tlorentz vector is equal to the HT'
    bkgdArray['myHT'] = bkgdArray.apply(lambda row: sum(row['selJet_pt']),axis=1)
    bkgdArray['myHTDiff'] = bkgdArray['HT']-bkgdArray['myHT']
    print bkgdArray['myHTDiff']
    plt.hist2d(bkgdArray['HT'],bkgdArray['myHT'],bins=[30,30])
    plt.show()

