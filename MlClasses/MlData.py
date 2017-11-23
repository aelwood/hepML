from sklearn.cross_validation import train_test_split

class MlData(object):
    '''Class for preparing the data provided in a dataframe for machine learning
    e.g. Used for splitting into test and evaluation sets'''
    def __init__(self,df,classifier):
        self.y = df[classifier]
        self.X = df.drop(classifier,axis=1)

    def split(self, evalSize=0.33, testSize=0.33):
        # (thanks Tim Head https://betatim.github.io/posts/sklearn-for-TMVA-users/ )
        self.X_dev,self.X_eval, self.y_dev,self.y_eval = \
                train_test_split(self.X, self.y, test_size=testSize, random_state=42)

        self.X_train,self.X_test, self.y_train,self.y_test = \
                train_test_split(self.X_dev, self.y_dev,test_size=evalSize, random_state=492)
        
