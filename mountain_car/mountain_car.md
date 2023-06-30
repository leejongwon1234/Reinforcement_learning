# 강화학습 - Mountain car
----------------------------------------------------

* 목차

>1.사고과정 및 함수 코드

>2.코드 구조

>3.고난과 고뇌, 결과
<pre>
</pre>

## 1. 사고과정 및 코드
<pre>
</pre>
![mountain car 설계도](https://github.com/leejongwon1234/-4-/assets/123047859/2746e1a8-956b-4fb9-a6ce-35dab516ee5b)

- 환경 구성

```python
env = gym.make("MountainCar-v0")
```

- 환경 초기화 함수 : 에피소드 시작할 떄마다
```python
state = env.reset()[0] #env.reset()은 (state,{})를 반환 #state는 크기가 2인 1차원 배열
state = state.reshape(1, 2) #state를 1x2인 2차원배열로 변환
```

- 특정 환경에서 action을 취했을 때 결과를 반환하는 함수
```python
env.step(action) : #(state, reward, done, info)를 반환
```

- 이 결과를 memory에 저장해둬서 교과서 만들기

```python
#memory에 저장할 때는 (state, action, reward, next_state)를 저장해야겠다.
  memory_b.append((state, action, reward, next_state, done))
```


- agent가 어떤 action을 취할 지 학습(Q)을 바탕으로 예측하는 함수
```python
action = np.argmax(np.array(model.call(state))[0])
```

- 학습을 바탕으로 예측값을 따르지 말고 EPSILON의 확률로 탐험, EPSION은 STEP마다 감소
```python
for i in range(8000): #에피소드 스텝

# e-greedy 점점 줄어들게 설정
  eps = np.exp(-i/2000)

  for t in range(200): # 타임 스텝

    # Inference: e-greedy
    if np.random.rand() < eps:
        action = np.random.randint(0, 3)
    else:
        action = np.argmax(np.array(model.call(state))[0])
```
- 리워드가 끝나기 전까지 항상 1이므로, 변경시켜 주자
```python
    next_state, reward, done = env.step(action)[0:3]
    next_state = next_state.reshape(1, 2)

    if done :
        reward = 1000
    elif next_state[0][0] >= 0.4:
        reward = 128
    elif next_state[0][0] >= 0.3:
        reward = 64
    elif next_state[0][0] >= 0.2:
        reward = 32
    elif next_state[0][0] >= 0.1:
        reward = 16
    elif next_state[0][0] >= 0.05:
        reward = 8
    elif next_state[0][0] >= 0:
        reward = 4
    elif next_state[0][0] >= -0.1:
        reward = 2
    elif next_state[0][0] >= -0.2:
        reward = 1
    elif next_state[0][0] >= -0.3:
        reward = 0
    else:
        reward = -1
```
- 여러번의 수행과정을 거치고, 데이터를 메모리에 저장했다면 이제 학습을 시키자.

```python
  if i > 64:
        minibatch = random.sample(memory_b, 128)

        states = np.array([x[0][0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch]) 
        next_states = np.array([x[3][0] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        target_y = model.call(states).numpy()
        target_y[range(128), actions] = rewards + 0.95 * np.max(model.call(next_states).numpy(), axis=1)

        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(target_y - model.call(states)))
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

- 만약 action을 했는데 done이 True라면 or 목표치 500step이 끝났다면, 에피소드를 종료하고 다시 초기화
```python
if done :
        print('Episode', i,'total reward ',total_reward,'Score', score,'Step', t)
        scores.append(score)
        break
elif t == 199:
        print('Episode', i,'total reward ',total_reward,'Score', score)
        scores.append(score)
        break
```
<pre>
</pre>
## 2. 코드
<pre>
</pre>
- 코드
```python
#import library
import gym
import numpy as np
import random
from collections import deque #저장 도와주는 라이브러리
from keras.layers import Dense
import tensorflow as tf
from keras import Model

class DQN(Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.d1 = Dense(32,activation ='tanh', input_dim = 2)
        self.d3 = Dense(16,activation ='tanh')
        self.d4 = Dense(3,activation ='linear')
        self.optimizer = tf.keras.optimizers.Adam(0.001)


    def call(self, x):
        x = self.d1(x)
        x = self.d3(x)
        x = self.d4(x)

        return x
    
model = DQN()

memory_b = deque(maxlen=5000)
# MountainCar 환경 구성
env = gym.make("MountainCar-v0")
#scores
scores = []

for i in range(8000): #에피소드 스텝
  state = env.reset()[0] #env.reset()은 (state,{})를 반환 #state는 크기가 2인 1차원 배열
  state = state.reshape(1, 2) #state를 1x2인 2차원배열로 변환

  # e-greedy 점점 줄어들게 설정
  eps = np.exp(-i/2000)
  score = 0
  total_reward=0

  for t in range(200): # 타임 스텝
    # Inference: e-greedy
    if np.random.rand() < eps:
        action = np.random.randint(0, 3)
    else:
        action = np.argmax(np.array(model.call(state))[0])
    
    #다음 상태 및 보상
    next_state, reward, done = env.step(action)[0:3]
    next_state = next_state.reshape(1, 2)

    if done :
        reward = 1000
    elif next_state[0][0] >= 0.4:
        reward = 128
    elif next_state[0][0] >= 0.3:
        reward = 64
    elif next_state[0][0] >= 0.2:
        reward = 32
    elif next_state[0][0] >= 0.1:
        reward = 16
    elif next_state[0][0] >= 0.05:
        reward = 8
    elif next_state[0][0] >= 0:
        reward = 4
    elif next_state[0][0] >= -0.1:
        reward = 2
    elif next_state[0][0] >= -0.2:
        reward = 1
    elif next_state[0][0] >= -0.3:
        reward = 0
    else:
        reward = -1


    memory_b.append((state, action, reward, next_state, done))
    state = next_state
    total_reward += reward
    score -= 1

    if done :
        print('Episode', i,'total reward ',total_reward,'Score', score,'Step', t)
        scores.append(score)
        break
    elif t == 199:
        print('Episode', i,'total reward ',total_reward,'Score', score)
        scores.append(score)
        break

  if i > 64:
        minibatch = random.sample(memory_b, 128)

        states = np.array([x[0][0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch]) 
        next_states = np.array([x[3][0] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        target_y = model.call(states).numpy()
        target_y[range(128), actions] = rewards + 0.95 * np.max(model.call(next_states).numpy(), axis=1)

        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(target_y - model.call(states)))
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#학습 결과
import matplotlib.pyplot as plt
plt.scatter(range(len(scores)),scores)
plt.show()
```

- 결과
![mountain 최종](https://github.com/leejongwon1234/-4-/assets/123047859/cd32f581-70d0-4c6e-9c21-b010c3616c04)

<pre>
</pre>
## 3. 고난과 고뇌 및 결과
<pre>
</pre>
>### 보상을 어떻게 처리 할 것인가?
1. 처음에는 보상을 -1로 설정했다. : 그럼 무조건 빨리 끝내는게 좋아서 빨리 끝내는 방법을 찾아낼 것이라고 생각했다. 
   - 학습을 전혀 하지 못했다.
<pre>
</pre>
2. 오른쪽으로 가면 갈수록 기하급수적으로 높은 값의 보상을 설정 했다. : 그럼 더 높이 올라가기 위해 노력할 것이라 생각했다.
   - 학습을 했고, 정상에도 도달했으나 빨리 끝낼수록 좋은 목표를 학습했다고는 하기 힘든 결과를 냈다.
![Mountain car 1차 결과](https://github.com/leejongwon1234/-4-/assets/123047859/a82e1d1f-0906-47bf-8d95-6d591b351621)
<pre>
</pre>
3. 따라서 시간에 따라 보상을 달리하는 방법을 시도해봤다. : 그럼 빠르게 보상을 얻으려고 노력할 것이라 생각했다.
   - overfitting문제인지, 보상 체계의 문제인지 모르겠으나 학습을 포기하는 모습을 보였다.
![mountain car 보상 시간에 따른 다른 보상 설정](https://github.com/leejongwon1234/-4-/assets/123047859/26b2daed-09f1-4a77-a3d0-04128ea0a1a6)
<pre>
</pre>
4. 따라서 2번의 경우 보상을 10000으로 크게하고 episode를 10000까지 학습시켰다.
    - overfitting 문제인지, 보상 체계의 문제인지 모르겠으나 학습을 포기하는 모습을 보였다.
![mountain car 은닉층 4층 보상10000](https://github.com/leejongwon1234/-4-/assets/123047859/75ecd1fb-b681-4c10-8232-f6fc045976c6)
<pre>
</pre>
5. 한번 끝까지 학습시켜 보기위해 학습을 에피소드 20000회로 늘렸고, 보상을 2000으로 늘려보았다.
    - 학습을 포기하는 모습을 보였다.
 
![2000보상,은닉3층,20000회](https://github.com/leejongwon1234/-4-/assets/123047859/cc548430-867b-4b85-9143-58b50a6e2944)
<pre>
</pre>
- #### 결론 : 따라서 2번의 경우가 우리의 목표에 가장 가까운 결과값을 냈지만, 한계를 보였다.
<pre>
</pre>
>### 가장 적절한 epsilon 값이 어떤 값인지 잘 모르겠다.
1. epsilon 값을 지수함수를 이용해 에피소드가 늘어날수록 감소하도록 설계했는데, 적절한 값인지 잘 모르겠다.
<pre>
</pre>
>### 은닉층의 노드수가 많을 수록 좋은 결과를 낼 것이라고 생각했는데, 그렇지 않았다.
1. 은닉층이 3개인 경우, 4개인 경우, 5개인 경우를 모두 시도해봤는데, 3개인 경우가 가장 좋은 결과를 냈다.

2. 은닉층의 노드 수가 (36,12,8,3)인 경우 보다 (24,8,3)인 경우가 더 좋은 결과를 냈다.