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
            
    def get_batch_critic(self,model_actor, model_critic, batchSize = 25):
        len_memory = len(self.memory)
        num_outputs_critic = model_critic.output_shape[-1]
        num_outputs_actor = model_actor.output_shape[-1]
        
        inputs= np.zeros((min(len_memory, batchSize), self.memory[0][0][0].shape[1],
                                  self.memory[0][0][0].shape[2]))
        inputs_2= np.zeros((min(len_memory, batchSize), num_outputs_actor))
        
        targets_critic = np.zeros((min(len_memory, batchSize), num_outputs_critic))
        
        for i,idx in enumerate(np.random.randint(0, len_memory, size = 
                                                 min(len_memory, batchSize))):
            current_state, action, reward, next_state = self.memory[idx][0]
            done = self.memory[idx][1]
            
            inputs[i]  = current_state
            inputs_2[i] = action
            if done:
                targets_critic[i] = reward
            else:
                targets_critic[i] = reward + self.gamma * model_critic.predict([next_state,model_actor.predict(next_state)])[0]
        
        inputs_2 = np.reshape(inputs_2, (min(len_memory, batchSize) , num_outputs_actor))
        return inputs, inputs_2, targets_critic
    
    def get_batch_actor(self, model_actor, model_critic, inputsState, inputsAction):
        num_outputs_actor = model_actor.output_shape[-1]
        batchshape0 = np.shape(inputsAction)[0]
        batchshape1 = np.shape(inputsAction)[1]
        targets_actor  = np.zeros((batchshape0, num_outputs_actor))
        shape0 = np.shape(inputsState[0])[0]
        shape1 = np.shape(inputsState[0])[1]
        for i in range(batchshape0):
            Critic_q = model_critic.predict([np.reshape(inputsState[i], 
                                                           (1,shape0,shape1)), 
                                                np.reshape(inputsAction[i], 
                                                           (1,batchshape1))])[0]
            
            targets_actor[i]  = model_actor.predict(np.reshape(inputsState[i], (1,shape0,shape1)))[0]

            for j in range(num_outputs_actor):
                if targets_actor[i][j] < 0:
                    targets_actor[i][j] = targets_actor[i][j] - Critic_q
                else:
                    targets_actor[i][j] = targets_actor[i][j] + Critic_q

        return targets_actor
        