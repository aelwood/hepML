#Script to order the mean squared error from an output directory containing regression results
# e.g. python scripts/listRegression.py testPlots/mlPlots/regression/fourVectorMT2W
import sys
import glob
import os
import ast

if len(sys.argv)!=2: 
    exit('Usage: python listRegression.py <inputDir>')

inDir=sys.argv[1]

test={}
train={}

for d in glob.glob(os.path.join(inDir,'*')):
    name=d.split('/')[-1]

    f=open(os.path.join(d,'classificationReport.txt'))

    reachedTest=False
    for l in f.readlines():
        if '[' in l:
            if not reachedTest:
                test[name]=ast.literal_eval(l.strip())[0]
                reachedTest=True
            else:
                train[name]=ast.literal_eval(l.strip())[0]

print ''
print 'Test set results:'
for w in sorted(test, key=test.get):
    print test[w],'\t', w
print ''


print 'Train set results:'
for w in sorted(train, key=train.get):
    print train[w],'\t', w
print ''

