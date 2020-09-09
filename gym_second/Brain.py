#Brain class
#Class brain for - CartPole-v0
import keras
from keras.layers import Dense, Dropout, Conv1D,Conv2D, MaxPooling1D, Flatten
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras import metrics

class Brain():
    def __init__(self, IS = (6, 4), lr = 0.0001):
        self.InputShape = IS
        self.Learning_rate = lr
        self.num_outputs = 2
        
        self.model = Sequential()
        self.model.add(Conv1D(8, (2,),activation='relu',
                             input_shape = self.InputShape, name = 'conv_input'))
        self.model.add(Conv1D(16, (2,), activation='relu'))
        self.model.add(MaxPooling1D())
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        #self.model.add(Dropout(0.1))
        self.model.add(Dense(units = self.num_outputs))
        
        self.model.compile(loss = 'mse',
                           optimizer=Adam(learning_rate=self.Learning_rate))
        
    #Load_model function
    def load_model(self, filepath):
        self.model = load_model(filepath)
        return self.model
