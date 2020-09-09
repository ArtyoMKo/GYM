#Training file for - CartPole-v0
import numpy as np
import gym
from DQN_2 import Dqn
from BRAIN_2 import Brain

#Initializing the parameters
#1
memory_size = 100000
batch_size = [25]
learning_rate = [0.00015]
epsilion = 0
epsilion_delta = 0
min_epsilion = 0
gamma = 0.9
n_last_states = [2]
policy_noise = 0.1
nb_models = 1

env = [gym.make('BipedalWalker-v3') for _ in range(nb_models)]

max_action = env[0].action_space.high[0]
shapeOfState = env[0].observation_space.shape[0]
shapeOfAction = env[0].action_space.shape[0]
action = [env[i].action_space.sample() for i in range(len(env))]
index = [[0 for i in action[j]] for j in range(nb_models)]
idx = 2
brain = [Brain((shapeOfState, n_last_states[i]), learning_rate[i]) for i in range(nb_models)]

Dqn = [Dqn(memory_size, gamma) for _ in range(nb_models)]

filepathToSave = ['BipedalWalker-v3'+str(i)+'.h5' for i in range(nb_models)]

model = [brain[0].model for i in range(nb_models)]

model[0].summary()
#Reset function for states and environment
def resetStates(x):
    currentState = np.zeros(( 1, shapeOfState, n_last_states[x]))
    for i in range(n_last_states[x]):
        currentState[:,:,i] = env[x].reset()
    return currentState, currentState

epoch = 0
step_faile = 10
n = 0

while epoch < 5000:
    
    epoch +=1
    print(epoch)
    st = np.array([resetStates(k) for k in range(len(env))])
    currentState, nextState = st[:,0], st[:,1]
    steps = [0 for _ in env]
    Done = [False for _ in env]
    st_reward = [0 for _ in env]
    done = False
    tim = 0
    
    while done == False:
        
        for j in range(len(env)):
            if Done[j] == False:
                #env[j].render()
                if np.random.randn() < epsilion:
                    action[j] = env[j].action_space.sample()
                else:
                    Q_values = np.reshape(model[j].predict(currentState[j]), (shapeOfAction, idx))
                    
                    for i in range(shapeOfAction):
                        index[j][i] = np.argmax(Q_values[i])
                        if index[j][i] == 0:
                            action[j][i] = -1 * np.max(Q_values[i])
                        else:
                            action[j][i] = np.max(Q_values[i])
                        
                    action[j] += np.random.normal(0, policy_noise,size=shapeOfAction)
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
                steps[j] += 1
                state, reward, Done[j] ,_ = env[j].step(action[j])
                st_reward[j] += reward

                
                st = nextState[j]
                state = np.reshape(state, (1, shapeOfState, 1))
                st = np.append(st, state, axis = 2)
                st = np.delete(st, 0, axis = 2)
                nextState[j] = st
                
                #if steps > 50 or Done == True:
                Dqn[j].remember([currentState[j], index[j], reward, nextState[j]], Done[j])
                currentState[j] = nextState[j]
                done = True
                tim += 1
                
        for j in range(nb_models):
            done = done * Done[j]
    for t in range(tim):    
        for j in range(len(env)):
            inputs, targets = Dqn[j].get_batch(model[j], batch_size[j])
            model[j].train_on_batch(inputs, targets)
    print('Model: ' + str(j) + ' Epoch: ' + str(epoch) + ' Current steps: ' + str(steps[j])
    + ' Epsilion: {:.5f}'.format(epsilion) + ' tot reward: ' + str(st_reward[j]))
    if epsilion > min_epsilion:
        epsilion -= epsilion_delta
            
for i in range(len(env)):
    model[i].save(filepathToSave[i])
    env[i].close()
