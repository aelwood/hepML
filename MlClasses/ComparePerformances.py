from MlClasses.PerformanceTests import rocCurve

class ComparePerformances(object):
    def __init__(self,modelResults,output=None):
        self.modelResults = modelResults
        self.output = output

    def compareRoc(self,selectedResults=None,append=''):

        if selectedResults:
            results = {k:self.modelResults[k] for k in selectedResults+['truth']}
        else:
            results = self.modelResults

        rocCurve(results,output=self.output,append=append)
        pass
