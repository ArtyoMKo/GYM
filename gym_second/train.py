#Training file for - CartPole-v0
import numpy as np
import gym
from DQN import Dqn
from Brain import Brain

#Initializing the parameters
memory_size = 5000
batch_size = 25
learning_rate = 0.0007
epsilion = .2
epsilion_delta = 0.001
min_epsilion = 0.06
gamma = 0.9
n_last_states = 3

#Initializing environment, brain, model and dqn algorithm
env = gym.make('CartPole-v0')
shapeOfState = env.observation_space.shape[0]
brain = Brain((shapeOfState, n_last_states), learning_rate)

Dqn = Dqn(memory_size, gamma)

filepathToSave = 'CartPole-v0.h5'

model = brain.model

#Reset function for states and environment
def resetStates():
    currentState = np.zeros(( 1, shapeOfState, n_last_states))
    for i in range(n_last_states):
        currentState[:,:,i] = env.reset()
    return currentState, currentState

epoch = 0
early_stop = 100
n = 0

while True:
    currentState, nextState = resetStates()
    GameOver = False
    epoch +=1
    steps = 0
    Done = False
    
    while Done == False:
        env.render()
        if np.random.rand() < epsilion:
            action = env.action_space.sample()
        else:
            Q_values = model.predict(currentState)
            action = np.argmax(Q_values)
        steps += 1
        state, reward, Done ,_ = env.step(action)
        
        state = np.reshape(state, (1, shapeOfState, 1))
        nextState = np.append(nextState, state, axis = 2)
        nextState = np.delete(nextState, 0, axis = 2)
    
        Dqn.remember([currentState, action, reward, nextState], Done)
        currentState = nextState

        inputs, targets = Dqn.get_batch(model, batch_size)
        model.train_on_batch(inputs, targets)

    print('Epoch: ' + str(epoch) + ' Current steps: ' + str(steps)
    + ' Epsilion: {:.5f}'.format(epsilion))
    if epsilion > min_epsilion:
        epsilion -= epsilion_delta
env.close()
