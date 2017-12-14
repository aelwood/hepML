from keras.models import Sequential
from keras.layers import Dense,Dropout

def findLayerSize(layer,refSize):

    if isinstance(layer, float):
        return int(layer*refSize)
    elif isinstance(layer, int):
        return layer
    else:
        print 'WARNING: layer must be int or float'
        return None

def createDenseModel(inputSize=None,outputSize=None,hiddenLayers=[1.0],dropOut=None,activation='relu',optimizer='adam'):

    #check inputs are ok
    assert inputSize and outputSize, 'Must provide non-zero input and output sizes'
    assert len(hiddenLayers)>=1, 'Need at least one hidden layer'

    refSize=inputSize+outputSize

    model = Sequential()

    #Add the first layer, taking the inputs
    model.add(Dense(units=findLayerSize(hiddenLayers[0],refSize), 
        activation=activation, input_dim=inputSize,name='input'))

    if dropOut: model.add(Dropout(dropOut))

    #Add the extra hidden layers
    for layer in hiddenLayers[1:]:
        model.add(Dense(units=findLayerSize(hiddenLayers[0],refSize), 
            activation=activation))

        if dropOut: model.add(Dropout(dropOut))

    #Add the output layer and choose the type of loss function
    #Choose the loss function based on whether it's binary or not
    if outputSize==2: 
        #It's better to choose a sigmoid function and one output layer for binary
        # This is a special case of n>2 classification
        model.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    else: 
        #Softmax forces the outputs to sum to 1 so the score on each node
        # can be interpreted as the probability of getting each class
        model.add(Dense(outputSize, activation='softmax'))
        loss = 'categorical_crossentropy'

    #After the layers are added compile the model
    model.compile(loss=loss,
        optimizer=optimizer,metrics=['accuracy'])

    return model


