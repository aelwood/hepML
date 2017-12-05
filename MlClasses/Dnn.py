from keras.models import Sequential
from keras.layers import Dense
from MlClasses.PerformanceTests import rocCurve


class Dnn(object):

    def __init__(self,data,output=None):
        self.data = data
        self.output = output

    def setup(self,hiddenLayers=[]):

        '''Setup the neural net. Input a list of hiddenlayers
        if you fill float takes as fraction of inputs+outputs
        if you fill int takes as number. 
        E.g.:  hiddenLayers=[0.66,20] 
        has two hidden layers, one with 2/3 of input plus output 
        neurons, second with 20 neurons.
        ''' 

        self.model = Sequential()
        self.model.add(Dense(units=64, activation='relu', 
            input_dim=len(self.data.X_train.columns)))
        self.model.compile(loss='categorical_crossentropy',
            optimizer='sgd',metrics=['accuracy'])

    def fit(self):

        self.model.fit(self.data.X_train, self.data.Y_train, 
                epochs=5, batch_size=32)

    def classificationReport(self):

        self.report = model.evaluate(self.data.X_test, self.data.Y_test, batch_size=128)

        if not os.path.exists(self.output): os.makedirs(self.output)
        f=open(os.path.join(self.output,'classificationReport.txt'),'w')
        f.write( 'Performance on test set:')
        f.write(self.report)

    def rocCurve(self):

        rocCurve(self.model.predict(self.data.X_test), batch_size=128),self.data.X_test,self.data.y_test,self.output)
        rocCurve(self.model.predict(self.data.X_test),self.data.X_train,self.data.y_train,self.output,append='_train')

    def diagnostics(self):
        self.classificationReport()
        self.rocCurve()

