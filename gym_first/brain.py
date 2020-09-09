#Class brain for - acrobat
import keras
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, concatenate
from keras.optimizers import Adam
from keras.models import Sequential, load_model

class Brain():
    def __init__(self, IS = (6, 4), lr = 0.0001):
        self.InputShape = IS
        self.Learning_rate = lr
        self.num_outputs = 3
        
        state_Input = keras.Input(self.InputShape, name='Input_1')
        x1 = Conv1D(16, (3,),activation='relu',
                              name = 'conv_input')(state_Input)
        x1 = MaxPooling1D()(x1)
        x2 = Conv1D(32, (2,),activation='relu',
                             name = 'conv_2')(x1)
        x2 = Flatten()(x2)
        x3 = Dense(32, activation='relu')(x2)
        x4 = Dense(64, activation='relu')(x3)
        x5 = Dense(9, activation='relu')(x4)
        outputOne = Dense(units = self.num_outputs)(x5)
        
        #y1 = Conv1D(16, (3,),activation='relu',
         #                     name = 'conv_input')(state_Input)
        #y1 = MaxPooling1D()(y1)
        #y2 = Conv1D(32, (2,),activation='relu',
        #                     name = 'conv_2')(y1)
        #y2 = Flatten()(y2)
        #conc = concatenate([y2, outputOne])
        #z1 = Dense(256, activation='relu')(conc)
        #z2 = Dense(units = self.num_outputs)(z1)
        
        #self.model.add(Conv1D(16, (3,),activation='relu',
        #                     input_shape = self.InputShape, name = 'conv_input'))
        #self.model.add(MaxPooling1D())
        #self.model.add(Conv1D(32, (2,),activation='relu',
        #                     name = 'conv_2'))
        #self.model.add(Flatten())
        #self.model.add(Dense(256, activation='relu'))
        
        #self.model.add(Dropout(0.1))
        #self.model.add(Dense(64, activation='relu'))
        #outputOne = self.model.add(Dense(units = self.num_outputs))
        self.model = keras.Model(inputs = state_Input, outputs = outputOne)
        self.model.compile(loss = 'mse',
                           optimizer=Adam(learning_rate=self.Learning_rate))
        #self.model.compile(loss = 'mse',
        #                   optimizer=Adam(learning_rate=self.Learning_rate))

    #Load_model function
    def load_model(self, filepath):
        self.model = load_model(filepath)
        return self.model
    
    def save_img(self,model, filepath):
        keras.utils.plot_model(model, filepath, show_shapes=True)
        return 0
    
    def callbacks(self, _patience):
        callbacks = [
            keras.callbacks.EarlyStopping(
                # Прекратить обучение если `val_loss` больше не улучшается
                monitor='val_loss',
                # "больше не улучшается" определим как "не лучше чем 1e-2 и меньше"
                min_delta=1e-2,
                # "больше не улучшается" далее определим как "как минимум в течение * эпох"
                patience=_patience,
                verbose=1)
        ]
        return callbacks