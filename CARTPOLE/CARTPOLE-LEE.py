import gymnasium as gym
import numpy as np
import random
from collections import deque #저장 도와주는 라이브러리
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# 뉴럴 네트워크 모델 만들기
model = Sequential()
model.add(Dense(48, input_dim=4, activation='tanh'))
model.add(Dense(24, activation='tanh'))
model.add(Dense(2, activation='linear'))
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

score = []
memory_b = deque(maxlen=3000) #리플레이 버퍼 저장

# CartPole 환경 구성
env = gym.make('CartPole-v1')

for i in range(1000): #에피소드 스텝

  state = env.reset()[0] #env.reset()은 (state,{})를 반환
  state = state.reshape(1, 4)
  # e-greedy 점점 줄어들게 설정
  eps = (2/10)* np.exp(-i/100)

  for t in range(500): # 타임 스텝

    # Inference: e-greedy
    if np.random.rand() < eps:
        action = np.random.randint(0, 2)
    else:
        action = np.argmax(model.predict(state,verbose=0))
    
    #다음 상태 및 보상
    next_state, reward, done = env.step(action)[0:3]

    next_state = next_state.reshape(1, 4)

    memory_b.append((state, action, reward, next_state, done))
    state = next_state

    if done or t == 499:
        print('Episode', i, 'Score', t + 1)
        score.append(t + 1)
        break
      

  if i > 20:
        minibatch = random.sample(memory_b, 32)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + 0.95 * np.amax(model.predict(next_state,verbose=0)[0])
            target_o = model.predict(state,verbose=0)
            target_o[0][action] = target
            model.fit(state, target_o, epochs=1, verbose=0) 

env.close()
print(score)