#Brain class
#Class brain for - CartPole-v0
import keras
from keras.layers import Dense, Dropout, Conv1D,Conv2D, MaxPooling1D, Flatten, Input, concatenate
from keras.optimizers import Adam
from keras.models import Sequential, load_model, Model
import keras.backend as K

class Brain():
    def __init__(self, IS = (24, 3), lr = 0.0001, _max_action = 1):
        self.InputShape = IS
        self.Learning_rate = lr
        self.num_outputs = 4
        self.max_action = _max_action
        
        inS = Input((self.InputShape), name='inputBrain')
        #Conv1 = Conv1D(128, (3,),activation='relu')(inS)
        x1 = Flatten()(inS)
        x2 = Dense(256, activation = 'relu', kernel_regularizer='l2')(x1)
        x2 = Dropout(0.3)(x2)
        x3 = Dense(512, activation = 'relu', kernel_regularizer='l2')(x2)
        out = Dense(4, activation = 'tanh')(x3)
        
        self.model_action = Model(inS , out)
        
        self.model_action.compile(loss = 'mae',
                           optimizer = Adam(learning_rate=self.Learning_rate))
        
        
    
    #Load_model function
    def load_model(self, filepath):
        self.model = load_model(filepath)
        return self.model
    
class Critic():
    def __init__(self, IS_1 = (24, 2), IS_2 = (4,), lr = 0.0001):
        self.InputShapeState = IS_1
        self.Learning_rate = lr
        self.InputShapeAction = IS_2
        
        input1 = Input((self.InputShapeState), name = 'inputState')
        input2 = Input((self.InputShapeAction), name= 'inputAction')
        flatt1 = Flatten()(input1)
        
        x1 = Dense(256, activation='relu', kernel_regularizer='l2')(concatenate([flatt1, input2]))
        x1 = Dropout(0.3)(x1)
        x2 = Dense(512, activation = 'relu', kernel_regularizer='l2')(x1)
        x3 = Dense(1)(x2)
        
        self.model = Model([input1,input2], x3)
        self.model.compile(loss = 'mse', 
                           optimizer = Adam(learning_rate=self.Learning_rate))
