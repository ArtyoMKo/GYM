#Class brain-2
import keras
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Input, Flatten, concatenate
from keras.optimizers import Adam
from keras.models import Model, load_model

class Brain():
    def __init__(self, iS = (24, 3), lr = 0.00001):
        self.inputShape = iS
        self.learningRate = lr
        self.numOutputs = 11
        
        x = Input(self.inputShape)
        #x1 = Conv1D(32,(3,), activation= 'relu')(x)
        #c2 = Conv1D(64,(2,), activation= 'relu')(x1)
        #p1 = MaxPooling1D()(c2)
        x2 = Flatten()(x)
        x2 = Dropout(0.1)(x2)
        x4 = Dense(512, activation = 'relu')(x2)
        
        x3 = Dense(256, activation='relu')(x4)
        #x3 = Dropout(0.1)(x3)
        y1 = Dense(2, activation='sigmoid')(x3)
        y2 = Dense(2, activation='sigmoid')(x3)
        y3 = Dense(2, activation='sigmoid')(x3)
        y4 = Dense(2, activation='sigmoid')(x3)
        
        
        self.model = Model(x, outputs = [y1,y2,y3,y4])
        
        self.model.compile(loss = 'mse',
                           optimizer = Adam(learning_rate = self.learningRate))
        
    def load_model(self, filepath):
        self.model = load_model(filepath)
        return self.model
        