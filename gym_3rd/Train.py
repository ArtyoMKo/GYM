#Training file for - CartPole-v0
import numpy as np
import gym
from DQN import Dqn
from BRAIN import Brain, Critic

#Initializing the parameters
memory_size = 50000
batch_size = 50
learning_rate = 0.001
epsilion = .1
epsilion_delta = 0.001
min_epsilion = 0.001
gamma = 0.99
n_last_states = 2
policy_noise = 0.1

#Initializing environment, brain, model and dqn algorithm
env = gym.make('BipedalWalker-v3')

max_action = env.action_space.high[0]
shapeOfState = env.observation_space.shape[0]
shapeOfAction = env.action_space.shape[0]

brain = Brain((shapeOfState, n_last_states), learning_rate, max_action)

Dqn = Dqn(memory_size, gamma)

filepathToSave = 'BipedalWalker-v3.h5'

model_actor = brain.model_action

critic = Critic((shapeOfState, n_last_states), (shapeOfAction,), learning_rate)
model_critic = critic.model

num_outAct = model_actor.output_shape[-1]
model_actor.summary()
model_critic.summary()
#Reset function for states and environment
def resetStates():
    currentState = np.zeros(( 1, shapeOfState, n_last_states))
    for i in range(n_last_states):
        currentState[:,:,i] = env.reset()
    return currentState, currentState

epoch = 0
step_faile = 10
n = 0

while epoch < 5000:
    currentState, nextState = resetStates()
    GameOver = False
    epoch +=1
    steps = 0
    Done = False
    st_reward = 0
    tim = 0
    while Done == False:
        #env.render()
        if np.random.randn() < epsilion:
            action = env.action_space.sample()
        else:
            action = model_actor.predict(currentState)[0] * max_action
            action += np.random.normal(0, policy_noise,size=num_outAct)
            action = action.clip(-1, 1)
        
        #print('action: ' + str(action))
        steps += 1
        state, reward, Done ,_ = env.step(action)
        st_reward += reward
        
        state = np.reshape(state, (1, shapeOfState, 1))
        nextState = np.append(nextState, state, axis = 2)
        nextState = np.delete(nextState, 0, axis = 2)
        
        Dqn.remember([currentState, action, reward, nextState], Done)
        currentState = nextState
        tim += 1
    
    
    inputsState, inputsAction, targets_critic = Dqn.get_batch_critic(model_actor, model_critic, batch_size)
    
    model_critic.train_on_batch(x = [inputsState,inputsAction], y = targets_critic)
    if tim % 3 == 0:
        targets_actor = Dqn.get_batch_actor(model_actor, model_critic, inputsState, inputsAction)
        
        model_actor.train_on_batch(inputsState, targets_actor)
        
    print('Epoch: ' + str(epoch) + ' Current steps: ' + str(steps)
    + ' Epsilion: {:.5f}'.format(epsilion) + ' tot reward: ' + str(st_reward))
    if epsilion > min_epsilion:
        epsilion -= epsilion_delta
        
model_actor.save(filepathToSave)
env.close()
