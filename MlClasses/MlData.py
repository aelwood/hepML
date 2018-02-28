import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class MlData(object):
    '''Class for preparing the data provided in a dataframe for machine learning
    e.g. Used for splitting into test and evaluation sets'''
    def __init__(self,df,classifier,weights=None):

        self.y = df[classifier]
        self.X = df.drop(classifier,axis=1)

        if weights is not None:
            assert len(weights)==len(self.y), 'Weights must be same size as input'

        self.weights=weights

        self.isSplit=False
        self.standardised=False #keep track of the train and test set standardisation
        self.standardisedDev=False #keep track of the development test standardisation

    def split(self, evalSize=0.0, testSize=0.33,limitSize=None):
        # (thanks Tim Head https://betatim.github.io/posts/sklearn-for-TMVA-users/ )
        # An evaluation set is useful when hyperparameters are optimised based on the performance 
        # of the test set, which allows information to leak from the test set. An evaluation
        # set then allows fully unbiased testing.
        
        if limitSize :
            #Allows the possibility to limit the size of the dataset, usually for testing
            if self.weights is None:
                self.X_dev,self.X_eval, self.y_dev,self.y_eval = \
                        train_test_split(self.X, self.y, test_size=int(evalSize*limitSize),
                                train_size=int((1-evalSize)*limitSize), random_state=42)
                self.weights_dev,self.weights_eval = None,None
            else:
                self.X_dev,self.X_eval, self.y_dev,self.y_eval, self.weights_dev,self.weights_eval = \
                        train_test_split(self.X, self.y, self.weights, test_size=int(evalSize*limitSize),
                                train_size=int((1-evalSize)*limitSize), random_state=42)
        else:
            if self.weights is None:
                self.X_dev,self.X_eval, self.y_dev,self.y_eval = \
                        train_test_split(self.X, self.y, test_size=evalSize, random_state=42)
                self.weights_dev,self.weights_eval = None,None
                
            else:
                self.X_dev,self.X_eval, self.y_dev,self.y_eval, self.weights_dev,self.weights_eval = \
                        train_test_split(self.X, self.y, self.weights, test_size=evalSize, random_state=42)

        if self.weights is None:
            self.X_train,self.X_test, self.y_train,self.y_test = \
                    train_test_split(self.X_dev, self.y_dev, test_size=testSize, random_state=492)
            self.weights_train,self.weights_test = None,None
        else:
            self.X_train,self.X_test, self.y_train,self.y_test, self.weights_train, self.weights_test = \
                    train_test_split(self.X_dev, self.y_dev, self.weights_dev, test_size=testSize, random_state=492)

        self.isSplit=True

        
    def removeDuplicates(self):
        '''Remove any duplicate columns by name'''

        if self.isSplit: print 'Warning: duplicates are removed after test/train/dev split, make sure to split again'

        #Removes a list of duplicated names
        self.X = self.X.loc[:,~self.X.columns.duplicated()]


    def standardise(self,standardiseDev=False): #Note that this does not change self.X or self.y or the development set
        '''Standardise the data to ensure the training occurs efficiently. 
        Normalises each dataset to have a mean of 0 and s.d. 1, weights are not taken into account...'''

        if self.isSplit==False: 
            print 'Warning: you must split the data before standardising.'
            print 'This avoids information leaking from the train to test set'
        
        if not self.standardised:
            self.scaler = preprocessing.StandardScaler().fit(self.X_train)
            self.X_train = pd.DataFrame(self.scaler.transform(self.X_train),columns=self.X_train.columns,index=self.X_train.index)
            self.X_test = pd.DataFrame(self.scaler.transform(self.X_test),columns=self.X_test.columns,index=self.X_test.index)

        if standardiseDev:
            self.X_dev = pd.DataFrame(self.scaler.transform(self.X_dev),columns=self.X_dev.columns,index=self.X_dev.index)
            self.X_eval = pd.DataFrame(self.scaler.transform(self.X_eval),columns=self.X_eval.columns,index=self.X_eval.index)
            self.standardisedDev=True


        self.standardised=True

    def unStandardise(self,justDev=False):
        '''Reverse the standardise operation'''
        if not justDev:
            self.X_train = pd.DataFrame(self.scaler.inverse_transform(self.X_train),columns=self.X_train.columns,index=self.X_train.index)
            self.X_test = pd.DataFrame(self.scaler.inverse_transform(self.X_test),columns=self.X_test.columns,index=self.X_test.index)
            self.standardised=False

        if self.standardisedDev:
            self.X_dev = pd.DataFrame(self.scaler.inverse_transform(self.X_dev),columns=self.X_dev.columns,index=self.X_dev.index)
            self.X_eval = pd.DataFrame(self.scaler.inverse_transform(self.X_eval),columns=self.X_eval.columns,index=self.X_eval.index)
            self.standardisedDev=False

        pass


    def prepare(self,evalSize=0.0, testSize=0.33,limitSize=None,standardise=True):
        '''Convenience function to remove duplicates, split and standardise'''
        self.removeDuplicates()
        self.split(evalSize=evalSize,testSize=testSize,limitSize=limitSize)
        if standardise: self.standardise() #This needs to be done after the split to not leak info into the test set



        

