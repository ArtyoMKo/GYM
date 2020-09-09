#Class for DQN algorithm - acrobat
import numpy as np

class DQN():
    def __init__(self, _max_memory, _gamma):
        self.max_memory = _max_memory
        self.gamma = _gamma
        self.memory = list()
        
    def remember(self, transition, done):
        self.memory.append([transition, done])
        if len(self.memory) > self.max_memory:
            del(self.memory[0])
            
    def get_batch(self, model, batchSize = 25):
        len_memory = len(self.memory)
        num_outputs = model.output_shape[-1]
        
        inputs = np.zeros((min(len_memory, batchSize), self.memory[0][0][0].shape[1],
                           self.memory[0][0][0].shape[2]))
        
        targets = np.zeros((min(len_memory, batchSize), num_outputs))
        
        for i,idx in enumerate(np.random.randint(0, len_memory, size = 
                                                 min(len_memory, batchSize))):
            current_state, action, reward, next_state = self.memory[idx][0]
            done = self.memory[idx][1]
            
            inputs[i]  = current_state
            targets[i] = model.predict(current_state)[0][1]
            Q_sa = np.max(model.predict(next_state)[0][1])
            
            if done:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.gamma * Q_sa
        
        return inputs, targets
    
            
        