from MlClasses.PerformanceTests import rocCurve

class ComparePerformances(object):
    def __init__(self,modelResults,output=None):
        self.modelResults = modelResults
        self.output = output

    def compareRoc(self):
        rocCurve(self.modelResults,output=self.output)
        pass
