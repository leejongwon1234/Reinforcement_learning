import gymnasium as gym
import numpy as np
import random
from collections import deque #저장 도와주는 라이브러리
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

#DQN 네트워크 모델
model = Sequential() 
model.add(Dense(48, input_dim=3, activation='ReLU'))
model.add(Dense(24, activation='ReLU'))
model.add(Dense(3, activation='linear'))
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

scores=[]
#memory buffer
memory = deque(maxlen = 1000)

#pendulum 환경 구성
env = gym.make('Pendulum-v1', g=1)
episode = 100
step = 200
for i in range(episode):

    state = env.reset()[0] #state = [x,y,angular velocity]
    state = state.reshape(1,3)
    eps = (4/10)* np.exp(-i/10)
    score = 0

    for t in range(step):

        #e-greedy
        if np.random.rand() < eps:
            action = np.random.choice([-2,0,2])
        else:
            action = int((np.argmax(model.predict(state,verbose=0))-1)*2)

        #다음 상태 및 보상
        next_state, reward = env.step((action,))[0:2]
        if next_state[0] > 0.95:
            score += 1
        next_state = next_state.reshape(1,3)

        memory.append((state, action, reward, next_state))
        state = next_state

    if i > 20:
        minibatch = random.sample(memory, 32)

        for state, action, reward, next_state in minibatch:
            action_to_index = int((action/2)+1)
            target = reward + 0.95 * np.amax(model.predict(next_state,verbose=0)[0])
            target_o = model.predict(state,verbose=0)
            target_o[0][action_to_index] = target
            model.fit(state, target_o, epochs=1, verbose=0)

    scores.append(score)
    print('episode : ', i, 'score : ', score, 'eps : ', eps)

env.close()

plt.scatter(np.arange(0,episode), scores)
plt.show()

