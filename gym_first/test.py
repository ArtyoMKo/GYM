#Testing file for - acrobat
import numpy as np
import gym
from Dqn import DQN
from brain import Brain

#Initializing the parameters
memory_size = 1000
batch_size = 25
#learning_rate = 0.0001
epsilion = .4
epsilion_delta = 0.0005
min_epsilion = 0.02
gamma = 0.9
n_last_states = 5

#Initializing environment, brain, model and dqn algorithm
env = gym.make('Acrobot-v1')
brain = Brain((6,n_last_states))
Dqn = DQN(memory_size, gamma)

filepathToLoad = 'acrobat.h5'

model = brain.load_model(filepathToLoad)

#Reset function for states and environment
def resetStates():
    currentState = np.zeros(( 1, 6, n_last_states))
    for i in range(n_last_states):
        currentState[:,:,i] = env.reset()
    return currentState, currentState

epoch = 0

while epoch < 5:
    currentState, nextState = resetStates()
    Done = False
    epoch +=1
    steps = 0
    while Done == False:
        env.render()
        if np.random.rand() > epsilion:
            Q_values = model.predict(currentState)
            action = np.argmax(Q_values)
        else:
            action = env.action_space.sample()
        steps += 1
        
        state, _, Done ,_ =env.step(action)
        state = np.reshape(state, (1, 6, 1))
        nextState = np.append(nextState, state, axis = 2)
        nextState = np.delete(nextState, 0, axis = 2)
        
        currentState = nextState
    print('Epoch: ' + str(epoch) + ' Current steps: ' + str(steps))
    
env.close()
