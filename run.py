from dfConvert import convertTree

makePlots=True

if __name__='__main__':

    #Load the dataframes
    signalFile = '/nfs/dust/cms/group/susy-desy/marco/training_sample_new/top_sample_0.root'
    bkgdFile = '/nfs/dust/cms/group/susy-desy/marco/training_sample_new/stop_sample_0.root'

    signal = convertTree(signalFile,signal=True,passFilePath=True)
    bkgd = convertTree(bkgdFile,signal=False,passFilePath=True)

    if makePlots:
        #make the plots using my pandas plotting stuff
        pass
