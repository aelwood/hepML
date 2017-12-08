import os

from MlClasses.PerformanceTests import rocCurve

class ComparePerformances(object):
    def __init__(self,models,output=None):
        self.models = models
        self.output = output

        if self.output:
            if not os.path.exists(self.output): os.makedirs(self.output)

    def compareRoc(self,selectedResults=None,append=''):

        if selectedResults:
            results = {k:self.models[k][0] for k in selectedResults+['truth']}
        else:
            results = {k:self.models[k][0] for k in self.models.keys()}

        rocCurve(results,output=self.output,append=append)
        pass

    def rankMethods(self):
        accuracies={}
        for name,model in self.models.iteritems():
            accuracies[name]=model[1]

        outFile = open(os.path.join(self.output,'accuracyRank.txt'),'w')

        for key, value in sorted(accuracies.iteritems(), key=lambda (k,v): (v,k)):
            outFile.write( "%s: %s\n" % (key, value))

        outFile.close()

