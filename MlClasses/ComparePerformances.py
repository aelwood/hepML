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
            results = {k:self.models[k].testPrediction() for k in selectedResults}
        else:
            results = {k:self.models[k].testPrediction() for k in self.models.keys()}

        #As the splitting is done with the same pseudorandom initialisation the test set is always the same
        results['truth']=self.models.values()[0].data.y_test

        rocCurve(results,output=self.output,append=append)
        pass

    def rankMethods(self):


        if len(self.models)==0: 
            print 'No models to process'
            return None
        
        for i,scoreType in enumerate(self.models.values()[0].scoreTypes):

            accuracies={}
            accuraciesCrossVal={}
            accuraciesCrossValError={}

            for name,model in self.models.iteritems():

                if scoreType not in model.scoreTypes: continue

                accuracies[name]=model.score[i+1]
                if model.crossValResults is not None:
                    accuraciesCrossVal[name]=model.crossValResults.mean()
                    accuraciesCrossValError[name]=model.crossValResults.std()
                else:
                    accuraciesCrossVal[name]=0.
                    accuraciesCrossValError[name]=0.

            outFile = open(os.path.join(self.output,scoreType+'Rank.txt'),'w')

            for key, value in sorted(accuracies.iteritems(), key=lambda (k,v): (v,k)):
                outFile.write( "%s: %s\n" % (key, value))

            outFile.close()

            outFile = open(os.path.join(self.output,'scoreRankCrossVal.txt'),'w')

            for key, value in sorted(accuraciesCrossVal.iteritems(), key=lambda (k,v): (v,k)):
                outFile.write( "%s: %s +/- %s\n" % (key, value, accuraciesCrossValError[key]))

            outFile.close()
