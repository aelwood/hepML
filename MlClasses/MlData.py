import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class MlData(object):
    '''Class for preparing the data provided in a dataframe for machine learning
    e.g. Used for splitting into test and evaluation sets'''
    def __init__(self,df,classifier):

        self.y = df[classifier]
        self.X = df.drop(classifier,axis=1)

        self.isSplit=False

    def split(self, evalSize=0.0, testSize=0.33):
        # (thanks Tim Head https://betatim.github.io/posts/sklearn-for-TMVA-users/ )
        # An evaluation set is useful when hyperparameters are optimised based on the performance 
        # of the test set, which allows information to leak from the test set. An evaluation
        # set then allows fully unbiased testing.
        
        if evalSize>0.0:
            self.X_dev,self.X_eval, self.y_dev,self.y_eval = \
                    train_test_split(self.X, self.y, test_size=evalSize, random_state=42)
        else:
            self.X_dev=self.X
            self.y_dev=self.y

        self.X_train,self.X_test, self.y_train,self.y_test = \
                train_test_split(self.X_dev, self.y_dev,test_size=testSize, random_state=492)

        self.isSplit=True

        
    def removeDuplicates(self):
        '''Remove any duplicate columns by name and warn if two columns are the same'''

        if self.isSplit: print 'Warning: duplicates are removed after test/train/dev split, make sure to split again'

        #Removes a list of duplicated names
        self.X = self.X.loc[:,~self.X.columns.duplicated()]


    def standardise(self):
        '''Standardise the data to ensure the training occurs efficiently. 
        Normalises each dataset to have a mean of 0 and s.d. 1'''

        if self.isSplit==False: 
            print 'Warning: you must split the data before standardising.'
            print 'This avoids information leaking from the train to test set'
        
        self.scaler = preprocessing.StandardScaler().fit(self.X_train)
        self.X_train = pd.DataFrame(self.scaler.transform(self.X_train),columns=self.X_train.columns,index=self.X_train.index)
        self.X_test = pd.DataFrame(self.scaler.transform(self.X_test),columns=self.X_test.columns,index=self.X_test.index)
        self.X_dev = pd.DataFrame(self.scaler.transform(self.X_dev),columns=self.X_dev.columns,index=self.X_dev.index)


    def unStandardise(self):
        '''Reverse the standardise operation'''
        self.X_train = pd.DataFrame(self.scaler.inverse_transform(self.X_train),columns=self.X_train.columns,index=self.X_train.index)
        self.X_test = pd.DataFrame(self.scaler.inverse_transform(self.X_test),columns=self.X_test.columns,index=self.X_test.index)
        self.X_dev = pd.DataFrame(self.scaler.inverse_transform(self.X_dev),columns=self.X_dev.columns,index=self.X_dev.index)
        pass


    def prepare(self,evalSize=0.0, testSize=0.33):
        '''Convenience function to remove duplicates, split and standardise'''
        self.removeDuplicates()
        self.split(evalSize=evalSize,testSize=testSize)
        self.standardise() #This needs to be done after the split to not leak info into the test set



        

