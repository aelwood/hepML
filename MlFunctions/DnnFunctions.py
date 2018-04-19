from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras import regularizers
from keras import backend as K

def findLayerSize(layer,refSize):

    if isinstance(layer, float):
        return int(layer*refSize)
    elif isinstance(layer, int):
        return layer
    else:
        print 'WARNING: layer must be int or float'
        return None

def createDenseModel(inputSize=None,outputSize=None,hiddenLayers=[1.0],dropOut=None,l2Regularization=None,activation='relu',optimizer='adam',doRegression=False,loss=None,extraMetrics=[]):
    '''
    Dropout: choose percentage to be dropped out on each hidden layer (not currently applied to input layer)
    l2Regularization: choose lambda of the regularization (ie multiplier of the penalty)
    '''

    #check inputs are ok
    assert inputSize and outputSize, 'Must provide non-zero input and output sizes'
    assert len(hiddenLayers)>=1, 'Need at least one hidden layer'

    refSize=inputSize+outputSize

    #Initialise the model
    model = Sequential()

    if l2Regularization: 
        regularization=regularizers.l2(l2Regularization)
    else:
        regularization=None

    #Add the first layer, taking the inputs
    model.add(Dense(units=findLayerSize(hiddenLayers[0],refSize), 
        activation=activation, input_dim=inputSize,name='input',
        kernel_regularizer=regularization))


    if dropOut: model.add(Dropout(dropOut))

    #Add the extra hidden layers
    for layer in hiddenLayers[1:]:
        model.add(Dense(units=findLayerSize(hiddenLayers[0],refSize), 
            activation=activation,kernel_regularizer=regularization))

        if dropOut: model.add(Dropout(dropOut))

    if not doRegression: # if doing a normal classification model

        #Add the output layer and choose the type of loss function
        #Choose the loss function based on whether it's binary or not
        if outputSize==2: 
            #It's better to choose a sigmoid function and one output layer for binary
            # This is a special case of n=2 classification
            model.add(Dense(1, activation='sigmoid'))
            if not loss: loss = 'binary_crossentropy'
        else: 
            #Softmax forces the outputs to sum to 1 so the score on each node
            # can be interpreted as the probability of getting each class
            model.add(Dense(outputSize, activation='softmax'))
            if not loss: loss = 'categorical_crossentropy'

        #After the layers are added compile the model
        model.compile(loss=loss,
            optimizer=optimizer,metrics=['accuracy']+extraMetrics)

    else: # if training a regression add output layer with linear activation function and mse loss

        model.add(Dense(1))
        if not loss: loss='mean_squared_error'
        model.compile(loss=loss,
            optimizer=optimizer,metrics=['mean_squared_error']+extraMetrics)


    return model

def significanceLoss(expectedSignal,expectedBkgd):
    '''Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size'''


    def sigLoss(y_true,y_pred):
        #Continuous version:

        signalWeight=expectedSignal/K.sum(y_true)
        bkgdWeight=expectedBkgd/K.sum(1-y_true)

        s = signalWeight*K.sum(y_pred*y_true)
        b = bkgdWeight*K.sum(y_pred*(1-y_true))

        return -(s*s)/(s+b+K.epsilon()) #Add the epsilon to avoid dividing by 0

    return sigLoss

def significanceLossInvert(expectedSignal,expectedBkgd):
    '''Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size'''


    def sigLossInvert(y_true,y_pred):
        #Continuous version:

        signalWeight=expectedSignal/K.sum(y_true)
        bkgdWeight=expectedBkgd/K.sum(1-y_true)

        s = signalWeight*K.sum(y_pred*y_true)
        b = bkgdWeight*K.sum(y_pred*(1-y_true))

        return (s+b)/(s*s+K.epsilon()) #Add the epsilon to avoid dividing by 0

    return sigLossInvert

def significanceLoss2Invert(expectedSignal,expectedBkgd):
    '''Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size'''


    def sigLoss2Invert(y_true,y_pred):
        #Continuous version:

        signalWeight=expectedSignal/K.sum(y_true)
        bkgdWeight=expectedBkgd/K.sum(1-y_true)

        s = signalWeight*K.sum(y_pred*y_true)
        b = bkgdWeight*K.sum(y_pred*(1-y_true))

        return b/(s*s+K.epsilon()) #Add the epsilon to avoid dividing by 0

    return sigLoss2Invert

def significanceLossInvertSqrt(expectedSignal,expectedBkgd):
    '''Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size'''


    def sigLossInvert(y_true,y_pred):
        #Continuous version:

        signalWeight=expectedSignal/K.sum(y_true)
        bkgdWeight=expectedBkgd/K.sum(1-y_true)

        s = signalWeight*K.sum(y_pred*y_true)
        b = bkgdWeight*K.sum(y_pred*(1-y_true))

        return K.sqrt(s+b)/(s+K.epsilon()) #Add the epsilon to avoid dividing by 0

    return sigLossInvert



def significanceFull(expectedSignal,expectedBkgd):
    '''Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size'''


    def significance(y_true,y_pred):
        #Discrete version

        signalWeight=expectedSignal/K.sum(y_true)
        bkgdWeight=expectedBkgd/K.sum(1-y_true)

        s = signalWeight*K.sum(K.round(y_pred)*y_true)
        b = bkgdWeight*K.sum(K.round(y_pred)*(1-y_true))

        return s/K.sqrt(s+b+K.epsilon()) #Add the epsilon to avoid dividing by 0

    return significance


def asimovSignificanceLoss(expectedSignal,expectedBkgd,systematic):
    '''Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size'''


    def asimovSigLoss(y_true,y_pred):
        #Continuous version:

        signalWeight=expectedSignal/K.sum(y_true)
        bkgdWeight=expectedBkgd/K.sum(1-y_true)

        s = signalWeight*K.sum(y_pred*y_true)
        b = bkgdWeight*K.sum(y_pred*(1-y_true))
        sigB=systematic*b

        return -2*((s+b)*K.log((s+b)*(b+sigB*sigB)/(b*b+(s+b)*sigB*sigB+K.epsilon())+K.epsilon())-b*b*K.log(1+sigB*sigB*s/(b*(b+sigB*sigB)+K.epsilon()))/(sigB*sigB+K.epsilon())) #Add the epsilon to avoid dividing by 0

    return asimovSigLoss

def asimovSignificanceLossInvert(expectedSignal,expectedBkgd,systematic):
    '''Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size'''


    def asimovSigLossInvert(y_true,y_pred):
        #Continuous version:

        signalWeight=expectedSignal/K.sum(y_true)
        bkgdWeight=expectedBkgd/K.sum(1-y_true)

        s = signalWeight*K.sum(y_pred*y_true)
        b = bkgdWeight*K.sum(y_pred*(1-y_true))
        sigB=systematic*b

        return 1./(2*((s+b)*K.log((s+b)*(b+sigB*sigB)/(b*b+(s+b)*sigB*sigB+K.epsilon())+K.epsilon())-b*b*K.log(1+sigB*sigB*s/(b*(b+sigB*sigB)+K.epsilon()))/(sigB*sigB+K.epsilon()))) #Add the epsilon to avoid dividing by 0

    return asimovSigLossInvert

def asimovSignificanceFull(expectedSignal,expectedBkgd,systematic):
    '''Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size'''


    def asimovSignificance(y_true,y_pred):
        #Continuous version:

        signalWeight=expectedSignal/K.sum(y_true)
        bkgdWeight=expectedBkgd/K.sum(1-y_true)

        s = signalWeight*K.sum(K.round(y_pred)*y_true)
        b = bkgdWeight*K.sum(K.round(y_pred)*(1-y_true))
        sigB=systematic*b

        return K.sqrt(2*((s+b)*K.log((s+b)*(b+sigB*sigB)/(b*b+(s+b)*sigB*sigB+K.epsilon())+K.epsilon())-b*b*K.log(1+sigB*sigB*s/(b*(b+sigB*sigB)+K.epsilon()))/(sigB*sigB+K.epsilon()))) #Add the epsilon to avoid dividing by 0

    return asimovSignificance


def truePositive(y_true,y_pred):
    return K.sum(K.round(y_pred)*y_true) / (K.sum(y_true) + K.epsilon())

def falsePositive(y_true,y_pred):
    return K.sum(K.round(y_pred)*(1-y_true)) / (K.sum(1-y_true) + K.epsilon())
