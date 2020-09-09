#Training file for - acrobat
import numpy as np
import gym
from Dqn import DQN
from brain import Brain

#Initializing the parameters
memory_size = 5000
batch_size = 25
learning_rate = 0.01
epsilion = .9
epsilion_delta = 0.002
min_epsilion = 0.7
gamma = 0.8
n_last_states = 6

#Initializing environment, brain, model and dqn algorithm
env = gym.make('Acrobot-v1')
shapeOfState = env.observation_space.shape[0]
brain = Brain(( shapeOfState, n_last_states), learning_rate)

Dqn = DQN(memory_size, gamma)

filepathToSave = 'acrobat2.h5'
ModelImgSave = 'arajin.png'

model = brain.model
#model = brain.load_model(filepathToSave)
callbacks = brain.callbacks(15)

#Reset function for states and environment
def resetStates():
    currentState = np.zeros(( 1, shapeOfState, n_last_states))
    for i in range(n_last_states):
        currentState[:,:,i] = env.reset()
    return currentState, currentState

epoch = 0

while epoch < 1000:
    currentState, nextState = resetStates()
    GameOver = False
    epoch +=1
    steps = 0
    action = env.action_space.sample()
    env.step(action)
    while GameOver == False:
        env.render()
        if np.random.rand() < 1 - epsilion:
            action = env.action_space.sample()
        else:
            Q_values = model.predict(currentState)[0]
            action = np.argmax(Q_values)
        steps += 1
        state, reward, Done ,_ = env.step(action)
        if Done:
            GameOver = True

        
        if reward > -1:
            if steps > 300:
                reward = 0.5
            elif steps > 250 and steps < 300:
                reward = 1
            elif steps > 200 and steps < 250:
                reward = 2
            elif steps < 200:
                reward = 3
            model.save(filepathToSave)
            
        elif steps < 250:
            reward = -0.4
        
        state = np.reshape(state, (1, shapeOfState, 1))
        nextState = np.append(nextState, state, axis = 2)
        nextState = np.delete(nextState, 0, axis = 2)
    
        Dqn.remember([currentState, action, reward, nextState], GameOver)
        currentState = nextState
        inputs, targets = Dqn.get_batch(model, batch_size)
        loss = model.train_on_batch(inputs, targets)
        
    print('loss: ' + str(loss))
        
    
       
    print('Epoch: ' + str(epoch) + ' Current steps: ' + str(steps)
    + ' Epsilion: {:.5f}'.format(1-epsilion))
    if epsilion > min_epsilion:
        epsilion -= epsilion_delta
env.close()
