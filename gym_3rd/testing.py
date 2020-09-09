import numpy as np
import gym
from BRAIN_2 import Brain

env = [gym.make('BipedalWalker-v3'),gym.make('BipedalWalker-v3'),gym.make('BipedalWalker-v3')]

max_action = env[0].action_space.high[0]
shapeOfState = env[0].observation_space.shape[0]
shapeOfAction = env[0].action_space.shape[0]
action = [env[i].action_space.sample() for i in range(len(env))]
index = [[0 for i in action[0]],[0 for i in action[1]],[0 for i in action[2]]]
n_last_states = [3,3,3]
learning_rate = [0.0025,0.0025,0.0025]
index = [[0 for i in action[0]],[0 for i in action[1]],[0 for i in action[2]]]
idx = 2

brain = [Brain((shapeOfState, n_last_states[0]), learning_rate[0]),
         Brain((shapeOfState, n_last_states[1]), learning_rate[1]),
         Brain((shapeOfState, n_last_states[2]), learning_rate[2])]

filePathsTOpen = ['BipedalWalker-v3'+str(0)+'.h5','BipedalWalker-v3'+str(1)+'.h5',
                  'BipedalWalker-v3'+str(2)+'.h5']

model = [brain[0].load_model(filePathsTOpen[0]),brain[1].load_model(filePathsTOpen[1]),
         brain[2].load_model(filePathsTOpen[2])]

def resetStates(x):
    currentState = np.zeros(( 1, shapeOfState, n_last_states[x]))
    for i in range(n_last_states[x]):
        currentState[:,:,i] = env[x].reset()
    return currentState, currentState

epoch = 0

while epoch < 1:
    epoch += 1
    st = np.array([resetStates(k) for k in range(len(env))])
    currentState, nextState = st[:,0], st[:,1]
    st_reward = [0 for _ in env]
    Done = [False for _ in env]
    
    while Done[0] == False or Done[1] == False or Done[2] == False:
        
        for j in range(len(env)):
            if Done[j] == False:
                env[j].render()
                Q_values = np.reshape(model[j].predict(currentState[j]), (shapeOfAction, idx))
                
                for i in range(shapeOfAction):
                    index[j][i] = np.argmax(Q_values[i])
                    if index[j][i] == 0:
                        action[j][i] = -1 * np.max(Q_values[i])
                    else:
                        action[j][i] = np.max(Q_values[i])
                    
                #action[j] += np.random.normal(0, policy_noise,size=shapeOfAction)
                action[j] = action[j].clip(env[j].action_space.low, env[j].action_space.high)
     #           action = np.zeros(4)
    #            Q_values = np.reshape(Q_values, (action.shape[0], 2))
        #        for i,idx in enumerate(Q_values):
         #           arg = np.argmax([idx[0], idx[1]])
          #          if arg == 0:
           #             action[i] = -1 * idx[arg] 
            #        else:
             #           action[i] = idx[arg]
            
            #print('action: ' + str(action))
            state, reward, Done[j] ,_ = env[j].step(action[j])
            st_reward[j] += reward
            st = currentState[j]
            state = np.reshape(state, (1, shapeOfState, 1))
            st = np.append(st, state, axis = 2)
            st = np.delete(st, 0, axis = 2)
            currentState[j] = st
            
    for j in range(len(env)):    
        print('model: ' + str(j) + ' reward: ' + str(st_reward[j]))
