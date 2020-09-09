#Class for DQN algorithm - acrobat
import numpy as np

class Dqn():
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
        num_outputs = model.output_shape
        
        inputs = np.zeros((min(len_memory, batchSize), self.memory[0][0][0].shape[1],
                           self.memory[0][0][0].shape[2]))
        
        targets = [np.zeros((min(len_memory, batchSize)
                             ,np.shape(num_outputs)[1])) for i in range(np.shape(num_outputs)[0])]
        #targets = np.zeros((min(len_memory, batchSize), 1))
        
        for i,idx in enumerate(np.random.randint(0, len_memory, 
                                                 size = min(len_memory, batchSize))):
            
            current_state, index, reward, next_state = self.memory[idx][0]
            done = self.memory[idx][1]
            inputs[i]  = current_state
            target = model.predict(current_state)
            Q_sa = np.reshape(model.predict(next_state),(np.shape(num_outputs)[0],
                            np.shape(num_outputs)[1]))
            for j in range(np.shape(num_outputs)[0]):
                targets[j][i] = target[j]
                if done:
                    targets[j][i][index[j]] = reward
                else:
                    targets[j][i][index[j]] = reward - self.gamma * np.max(Q_sa[j])
                    targets[j][i] += abs(np.random.normal(0, 0.2,size=2))
            #target = model.predict(current_state)[0]
            #targets[i] = target.mean()
            
            #target = np.reshape(targets[i], (4, 2))
            
            #Q_sa = np.reshape(Q_sa, (4, 2))
            #Q_sa = Q_sa.mean()
            #targets[i] = np.reshape(target, (num_outputs,))
        return inputs, targets
    
            
        