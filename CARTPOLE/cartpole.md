# 강화학습 - cartpole
----------------------------------------------------

* 목차

>1.사고과정 및 함수 코드

>2.코드 구조와 결과

>3.고난과 고뇌 및 해결방법
<pre>
</pre>

## 1. 사고과정 및 코드
<pre>
</pre>
![KakaoTalk_20230403_231257185](https://user-images.githubusercontent.com/123047859/229536982-3f49f13b-9bb4-4902-aa38-5eda87934823.jpg)

- 우선, 환경 구성
<pre><code>
env = gym.make('CartPole-v1')
</code></pre>

- 환경 초기화 함수 : 에피소드 시작할 떄마다 호출하면 되겠군.
<pre><code>
env.reset() : #(state,{})를 반환, env.reset()[0]를 이용해야겠다.
</code></pre>

- 특정 환경에서 action을 취했을 때 결과를 반환하는 함수
<pre><code>
env.step(action) : #(state, reward, done, info)를 반환
</code></pre>

- 이 결과를 memory에 저장해둬서 교과서를 만들어 놔야 학습을 시킬 수 있겠다.
<pre><code>
#memory에 저장할 때는 (state, action, reward, next_state)를 저장해야겠다.
memory_b.append((state, action, reward, next_state, done))
</code></pre>

- agent가 어떤 action을 취할 지 학습(Q)을 바탕으로 예측하는 함수
<pre><code>
mode.predict(state) : #action을 반환 :[0action을 연산한 수치,1action을 연산한 수치]
</code></pre>

- 학습을 바탕으로 예측값을 따르지 말고 EPSILON의 확률로 탐험, EPSION은 STEP마다 감소
<pre><code>
for i in range(1000): #에피소드 스텝

# e-greedy 점점 줄어들게 설정
  eps = (2/10)* np.exp(-i/100)

  for t in range(500): # 타임 스텝

    # Inference: e-greedy
    if np.random.rand() < eps:
        action = np.random.randint(0, 2)
    else:
        action = np.argmax(model.predict(state,verbose=0))
</code></pre>

- 여러번의 수행과정을 거치고, 데이터를 메모리에 저장했다면 이제 학습을 시키자.

<pre><code>
if i > 20: #episode 20번 이상부터 학습 시작
        minibatch = random.sample(memory_b, 32)#수많은 데이터 중 32개 뽑기

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + 0.95 * np.amax(model.predict(next_state,verbose=0)[0]) # 벨만 방정식

            target_o = model.predict(state,verbose=0)
            target_o[0][action] = target #타겟값을 예측값에 넣어줌
            model.fit(state, target_o, epochs=1, verbose=0) #학습
</code></pre>

- 만약 action을 했는데 done이 True라면 or 목표치 500step이 끝났다면, 에피소드를 종료하고 다시 초기화
<pre><code>
if done or t == 499:
        print('Episode', i, 'Score', t + 1)
        score.append(t + 1)
        break
</code></pre>
<pre>
</pre>
## 2. 코드와 결과
<pre>
</pre>
- 코드
<pre><code>
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
#model의 구조,학습방법,학습률
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

score = []
memory_b = deque(maxlen=3000) #리플레이 버퍼 저장, 3000데이터까지만 저장하고 더 많아지면 옛날 데이터 삭제

# CartPole 환경 구성
env = gym.make('CartPole-v1')

for i in range(1000): #에피소드 스텝(1000번까지 학습)

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
</code></pre>

- 결과
![image](https://user-images.githubusercontent.com/80379900/111059100-5b5b4a00-84d8-11eb-8b1a-5b5b5b5b5b5b.png)

<pre>
</pre>
## 3. 고난과 고뇌 및 해결방법
<pre>
</pre>
>### 갑자기 에피소드 200쯤에서 score가 급격히 떨어진다.
 - 학습하는 과정이 부족했다고 판단해서 에피소드를 1000번으로 늘렸다.
 - epsilon 값이 급격히 줄어 들어서 그런건 아닐까 생각하여 epsilon함수를 조정 했다.
 - 은닉층의 노드수를 늘렸다.

>### 가장 적절한 epsilon 값이 어떤 값인지 잘 모르겠다.
 - action이 2개밖에 없기 때문에 큰 상관은 없다고 판단했지만 만약 action이 많아져 경우의 수가 많아진다면 적절한 epsilon 값을 찾기 어렵지 않을까 생각해봤다.
 - 스터디 중 0.2-0.3으로 대헌씨가 설명해줬던게 기억나서 지수함수로 epsilon 값을 조정했다.

>### 환경이 주어져서 만들기가 어렵지 않았으나 환경구성을 우리가 해야했을 때의 난이도가 수직상승할 것임을 걱정해봤다.
 - ㅜㅜ