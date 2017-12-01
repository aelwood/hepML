from keras.models import Sequential
from MlClasses.PerformanceTests import rocCurve


class Dnn(object):

    def __init__(self,data,output=None):
        self.data = data
        self.output = output

    def setup(self):

        self.model = Sequential()
        self.model.add(Dense(units=64, activation='relu', 
            input_dim=len(self.data.X_train.keys())))
        self.model.compile(loss='categorical_crossentropy',
            optimizer='sgd',metrics=['accuracy'])

    def fit(self):

        self.model.fit(self.data.X_train, self.data.Y_train, 
                epochs=5, batch_size=32)

    def classificationReport(self):
        self.report = model.evaluate(self.data.X_test, self.data.Y_test, batch_size=128)
        print self.report

    def rocCurve(self):
        rocCurve(self.model.predict(self.data.X_test), batch_size=128),self.data.X_test,self.data.y_test,self.output)
        rocCurve(self.model.predict(self.data.X_test),self.data.X_train,self.data.y_train,self.output,append='_train')

    def diagnostics(self):
        self.classificationReport()
        self.rocCurve()

