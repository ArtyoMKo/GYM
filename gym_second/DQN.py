#Class DQN for CartPole-v0
import numpy as np

class Dqn():
    def __init__(self, _max_memory, gamma):
        self.max_memory = _max_memory
        self.gamma = gamma
        self.memory = list()
        
    def remember(self, transition, done):
        self.memory.append([transition, done])
        if self.max_memory < len(self.memory):
            del(self.memory[0])
            
    def get_batch(self, model, batchSize):
        len_memory = len(self.memory)
        inputs = np.zeros((min(len_memory, batchSize), self.memory[0][0][0].shape[1],
                           self.memory[0][0][0].shape[2]))
        num_outputs = model.output_shape[-1]
        targets = np.zeros((min(len_memory, batchSize), num_outputs))
        
        for i,idx in enumerate(np.random.randint(0, len_memory,
                                                 size=min(len_memory, batchSize))):
            currentSate, action, reward, nextState = self.memory[idx][0]
            done = self.memory[idx][1]
            
            inputs[i] = currentSate
            targets[i]= model.predict(currentSate)[0]
            
            Q_sa = np.max(model.predict(nextState)[0])
            if done:
                targets[i, action] = 0
            else:
                targets[i, action] = reward + self.gamma * Q_sa
                
        return inputs, targets