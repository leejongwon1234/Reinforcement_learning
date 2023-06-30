#강화학습 - Inverted Pendulum
----------------------------------------------------
*목차

> ### 1. 사고과정

> ### 2. g=1

> ### 3. g=5

> ### 4. g=10

> ### 5. 시행착오 및 해결 방법
<pre>
</pre>

## 1. 사고과정
<pre>
</pre>
![inverted_pendulum_thinking](https://github.com/leejongwon1234/Reinforcement_learning/assets/123047859/78cecf45-1980-4ff8-8e0d-e6b337a65222)

여기서 가중치를 업데이트(저장)하는 기준은 마지막 x_position이 0.999보다 클 때이다.
<pre>
</pre>
## 2. g=1
<pre>
</pre>
 - 구현 코드
```python
import gymnasium as gym
import numpy as np
import random
from collections import deque #저장 도와주는 라이브러리
from keras.layers import Dense
from tensorflow import keras
from keras import Model
import tensorflow as tf
import matplotlib.pyplot as plt

# 뉴럴 네트워크 모델 만들기
class DQN(Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.d1 = Dense(128, input_dim=3, activation='ReLU')
        self.d2 = Dense(64, activation='ReLU')
        self.d3 = Dense(32, activation='ReLU')
        self.d4 = Dense(5, activation='linear')
        self.optimizer = keras.optimizers.Adam(0.001)

        self.M = []  # 리플레이 버퍼

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        return x

#pendulum 환경 구성
env = gym.make('Pendulum-v1', g=1)
model = DQN()

memory = deque(maxlen=4000) #메모리 버퍼 4000

for j in range(1):
    episode = 500 # episode 500
    step = 200 # step 200
    reward_mean_graph=[]
    last_x_position_graph=[]
    for i in range(episode):

        state = env.reset()[0] #state = [x,y,angular velocity]
        state = state.reshape(1,3) #state = 1x3
        eps = ((5-j)/10)* np.exp(-i/100)
        reward_mean = []
        last_x_position = 0

        for t in range(step):
            #e-greedy
            if np.random.rand() < eps:
                action = np.random.choice([-2,-1,0,1,2]) #discrete한 action
            else:
                action = np.argmax(model.call(state))-2 #discrete한 action
            #다음 상태 및 보상
            next_state, reward = env.step((action,))[0:2]
            if t == step-1:
                last_x_position = next_state[0]
            next_state = next_state.reshape(1,3)
            reward_mean.append(reward)

            memory.append((state, action, reward, next_state))
            state = next_state

        if i > 20:
            minibatch = random.sample(memory, 64) 
            states = np.array([x[0][0] for x in minibatch])
            actions = np.array([x[1] for x in minibatch])
            rewards = np.array([x[2] for x in minibatch]) 
            next_states = np.array([x[3][0] for x in minibatch])

            actions_to_index = np.array([k+2 for k in actions])

            target_y = model.call(states).numpy()
            target_y[range(64), actions_to_index] = rewards + 0.95 * np.max(model.call(next_states).numpy(), axis=1) 

            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.square(target_y - model.call(states)))
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        m = np.mean(reward_mean)
        reward_mean_graph.append(m)
        last_x_position_graph.append(last_x_position)
        if last_x_position > 0.999 : #last_x_postion이 0.999이상일때 가중치 저장
            model.save_weights('pendulum_g1.h5')
        print(f'episode : {i} reward_mean: {m:.3f} eps : {eps:.3f} last_x_position : {last_x_position:.3f}')


    plt.scatter(np.arange(0,episode), reward_mean_graph, label='reward_mean_scatter')
    reward_mean_graph = np.array(reward_mean_graph)
    rmgm = reward_mean_graph.reshape(int(episode/10),10)
    rmgm = np.mean(rmgm, axis=1)
    rmgm = rmgm.reshape(1,int(episode/10))[0]

    plt.plot(np.arange(0,episode,10), rmgm, color='red', linewidth=2.0, label='reward_mean')
    plt.plot(np.arange(0,episode), last_x_position_graph, color='green', linewidth=2.0, label='last_x_position')
    plt.legend(loc='lower right')
    plt.show()

env.close()
```

- 결과
![g=1_1try_500episode_4000memory_64mini](https://github.com/leejongwon1234/Reinforcement_learning/assets/123047859/7008d0c7-2ec5-482f-869d-bb09cf12e30a)


![g=1_1try_500episode_4000memory_64mini_out](https://github.com/leejongwon1234/Reinforcement_learning/assets/123047859/304979c3-7761-48fe-b26a-282617e0ed66)

마지막 x_position은 1로 수렴하고, reward평균은 0으로 수렴한다.

- 테스트
```python
import gym
import numpy as np
from collections import deque
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import tensorflow as tf

# 뉴럴 네트워크 모델 만들기
class DQN(Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.d1 = Dense(128, input_dim=3, activation='relu')
        self.d2 = Dense(64, activation='relu')
        self.d3 = Dense(32, activation='relu')
        self.d4 = Dense(5, activation='linear')

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        return x

# pendulum 환경 구성
env = gym.make('Pendulum-v1', g=1, render_mode='human')

model = DQN()
model(np.zeros((1, 3)))  # 모델 호출하여 변수 생성

# 저장된 모델 로드
model.load_weights('pendulum_g1.h5')

episode = 10  # 테스트할 에피소드 수
step = 200

for i in range(episode):
    state = env.reset()[0]
    state = state.reshape(1, 3)
    reward_mean = []
    last_x_position = 0

    for t in range(step):
        # 테스트 시에는 epsilon-greedy 방법을 사용하지 않습니다.
        # 최적의 액션을 선택하기 위해 argmax 함수를 사용합니다.
        action = np.argmax(model(state)) - 2

        # 다음 상태 및 보상
        next_state, reward = env.step((action,))[0:2]
        next_state = next_state.reshape(1, 3)
        if t == step - 1:
            last_x_position = next_state[0][0]
        reward_mean.append(reward)

        state = next_state

    print(f'Test Episode: {i} Avg. Reward: {np.mean(reward_mean):.3f} Last x position: {last_x_position:.3f}')

env.close()
```
- 테스트 결과

![g=1_1try_500episode_4000memory_64mini_test_out](https://github.com/leejongwon1234/Reinforcement_learning/assets/123047859/a1afdb20-9108-42ee-a62f-ffd41a7df3fb)

![g1_video](https://github.com/leejongwon1234/Reinforcement_learning/assets/123047859/1f4e2c76-4ec6-4cde-bfd1-d24deb1b1b4f)
<pre>
</pre>
## 3. g=5
<pre>
</pre>
g=1과 달리, episode500인 경우를 3번에 반복하여 학습한다.(try=3)

episode가 500만큼 끝날 때 마다 epsilon값은 0.1차이로 reset된다.

학습 경향을 파악하기 위해 last_x_potions을 10episode마다 평균값으로 그래프를 그린다.(강한 중력으로 swing을 통해 막대를 세우는 경우 reward평균이 0으로 수렴하지 않기 떄문이다.)
<pre>
</pre>
 - 구현 코드
```python
import gymnasium as gym
import numpy as np
import random
from collections import deque #저장 도와주는 라이브러리
from keras.layers import Dense
from tensorflow import keras
from keras import Model
import tensorflow as tf
import matplotlib.pyplot as plt

# 뉴럴 네트워크 모델 만들기
class DQN(Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.d1 = Dense(128, input_dim=3, activation='ReLU')
        self.d2 = Dense(64, activation='ReLU')
        self.d3 = Dense(32, activation='ReLU')
        self.d4 = Dense(5, activation='linear')
        self.optimizer = keras.optimizers.Adam(0.001)

        self.M = []  # 리플레이 버퍼

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        return x

#pendulum 환경 구성
env = gym.make('Pendulum-v1', g=5)
model = DQN()

memory = deque(maxlen=4000)

for j in range(3):#try=3
    episode = 500
    step = 200
    reward_mean_graph=[]
    last_x_position_graph=[]
    for i in range(episode):

        state = env.reset()[0] #state = [x,y,angular velocity]
        state = state.reshape(1,3) #state = 1x3
        eps = ((5-j)/10)* np.exp(-i/100) #epsilon -0.1
        reward_mean = []
        last_x_position = 0

        for t in range(step):
            #e-greedy
            if np.random.rand() < eps:
                action = np.random.choice([-2,-1,0,1,2])
            else:
                action = np.argmax(model.call(state))-2
            #다음 상태 및 보상
            next_state, reward = env.step((action,))[0:2]
            if t == step-1:
                last_x_position = next_state[0]
            next_state = next_state.reshape(1,3)
            reward_mean.append(reward)

            memory.append((state, action, reward, next_state))
            state = next_state

        if i > 20:
            minibatch = random.sample(memory, 64) 
            states = np.array([x[0][0] for x in minibatch])
            actions = np.array([x[1] for x in minibatch])
            rewards = np.array([x[2] for x in minibatch]) 
            next_states = np.array([x[3][0] for x in minibatch])

            actions_to_index = np.array([k+2 for k in actions])

            target_y = model.call(states).numpy()
            target_y[range(64), actions_to_index] = rewards + 0.95 * np.max(model.call(next_states).numpy(), axis=1)

            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.square(target_y - model.call(states)))
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        m = np.mean(reward_mean)
        reward_mean_graph.append(m)
        last_x_position_graph.append(last_x_position)
        if last_x_position > 0.999 :
            model.save_weights('pendulum_g5.h5')
        print(f'episode : {i} reward_mean: {m:.3f} eps : {eps:.3f} last_x_position : {last_x_position:.3f}')


    plt.scatter(np.arange(0,episode), reward_mean_graph, label='reward_mean_scatter')
    reward_mean_graph = np.array(reward_mean_graph)
    rmgm = reward_mean_graph.reshape(int(episode/10),10)
    rmgm = np.mean(rmgm, axis=1)
    rmgm = rmgm.reshape(1,int(episode/10))[0]
    plt.plot(np.arange(0,episode,10), rmgm, color='red', linewidth=2.0, label='reward_mean')
    plt.legend(loc='lower right')
    plt.show()

#last_x_position을 10episode마다 평균으로 경향성을 파악한다.
    last_x_postion_mean_graph = np.array(last_x_position_graph)
    last_x_position_mean_graph = last_x_postion_mean_graph.reshape(int(episode/10),10)
    last_x_position_mean_graph = np.mean(last_x_position_mean_graph, axis=1)
    last_x_position_mean_graph = last_x_position_mean_graph.reshape(1,int(episode/10))[0]
    plt.plot(np.arange(0,episode,10), last_x_position_mean_graph, color='green', linewidth=2.0, label='last_x_position')
    plt.legend(loc='lower right')
    plt.show()

env.close()
```

- 결과

### - try1
![g=5_try1_rewardmean](https://github.com/leejongwon1234/Reinforcement_learning/assets/123047859/57bbb31c-f8a0-4c15-ae06-707c213ac7c9)
![g=5_try1_last_xposition](https://github.com/leejongwon1234/Reinforcement_learning/assets/123047859/43102100-d00f-4205-b058-850e2453087a)

### - try2
![g=5_try2_rewardmean](https://github.com/leejongwon1234/Reinforcement_learning/assets/123047859/a4d075d6-6bf6-46e0-87e0-fd63e1d46f49)
![g=5_try2_xposition](https://github.com/leejongwon1234/Reinforcement_learning/assets/123047859/acf802ef-efa9-4b41-bedf-17dd417e2464)

### - try3
![g=5_try3_xposition](https://github.com/leejongwon1234/Reinforcement_learning/assets/123047859/8cf46013-534f-4d1b-8cc5-1aff263f23b3)
![g=5_try3_rewardmean](https://github.com/leejongwon1234/Reinforcement_learning/assets/123047859/73423ef1-561d-4cda-a9e8-4137fee1edfa)

try1,2에서 학습하는 경향성을 보이다, try3에서 수렴하는 경향을 보인다.

- 테스트
```python
import gym
import numpy as np
from collections import deque
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import tensorflow as tf

# 뉴럴 네트워크 모델 만들기
class DQN(Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.d1 = Dense(128, input_dim=3, activation='relu')
        self.d2 = Dense(64, activation='relu')
        self.d3 = Dense(32, activation='relu')
        self.d4 = Dense(5, activation='linear')

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        return x

# pendulum 환경 구성
env = gym.make('Pendulum-v1', g=5, render_mode='human') #g=5

model = DQN()
model(np.zeros((1, 3)))  # 모델 호출하여 변수 생성

# 저장된 모델 로드
model.load_weights('pendulum_g5.h5')

episode = 10  # 테스트할 에피소드 수
step = 200

for i in range(episode):
    state = env.reset()[0]
    state = state.reshape(1, 3)
    reward_mean = []
    last_x_position = 0

    for t in range(step):
        # 테스트 시에는 epsilon-greedy 방법을 사용하지 않습니다.
        # 최적의 액션을 선택하기 위해 argmax 함수를 사용합니다.
        action = np.argmax(model(state)) - 2

        # 다음 상태 및 보상
        next_state, reward = env.step((action,))[0:2]
        next_state = next_state.reshape(1, 3)
        if t == step - 1:
            last_x_position = next_state[0][0]
        reward_mean.append(reward)

        state = next_state

    print(f'Test Episode: {i} Avg. Reward: {np.mean(reward_mean):.3f} Last x position: {last_x_position:.3f}')

env.close()
```
- 테스트 결과

![g=5_test_out](https://github.com/leejongwon1234/Reinforcement_learning/assets/123047859/f413977b-fc99-44dd-ab4d-6fad27a4fe88)

![g5_video](https://github.com/leejongwon1234/Reinforcement_learning/assets/123047859/24072ace-c098-4a5b-a75a-cb3941e4ea70)

<pre>
</pre>
## 3. g=10
<pre>
</pre>
g=1,5와 달리, episode500인 경우를 6번에 반복하여 학습한다.(try=6)

<pre>
</pre>
```python
import gymnasium as gym
import numpy as np
import random
from collections import deque #저장 도와주는 라이브러리
from keras.layers import Dense
from tensorflow import keras
from keras import Model
import tensorflow as tf
import matplotlib.pyplot as plt

# 뉴럴 네트워크 모델 만들기
class DQN(Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.d1 = Dense(128, input_dim=3, activation='ReLU')
        self.d2 = Dense(64, activation='ReLU')
        self.d3 = Dense(32, activation='ReLU')
        self.d4 = Dense(5, activation='linear')
        self.optimizer = keras.optimizers.Adam(0.001)

        self.M = []  # 리플레이 버퍼

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        return x

#pendulum 환경 구성
env = gym.make('Pendulum-v1', g=10)
model = DQN()

memory = deque(maxlen=4000) #############################################

for j in range(6):
    episode = 500 #######################################################
    step = 200 #############################################################
    reward_mean_graph=[]
    last_x_position_graph=[]
    for i in range(episode):

        state = env.reset()[0] #state = [x,y,angular velocity]
        state = state.reshape(1,3) #state = 1x3
        eps = ((5-(j/3))/10)* np.exp(-i/100) ##########################################
        reward_mean = []
        last_x_position = 0

        for t in range(step):
            #e-greedy
            if np.random.rand() < eps:
                action = np.random.choice([-2,-1,0,1,2])
            else:
                action = np.argmax(model.call(state))-2
            #다음 상태 및 보상
            next_state, reward = env.step((action,))[0:2]
            if t == step-1:
                last_x_position = next_state[0]
            next_state = next_state.reshape(1,3)
            reward_mean.append(reward)

            memory.append((state, action, reward, next_state))
            state = next_state

        if i > 20:
            minibatch = random.sample(memory, 64) #############################################################
            states = np.array([x[0][0] for x in minibatch])
            actions = np.array([x[1] for x in minibatch])
            rewards = np.array([x[2] for x in minibatch]) 
            next_states = np.array([x[3][0] for x in minibatch])

            actions_to_index = np.array([k+2 for k in actions])

            target_y = model.call(states).numpy()
            target_y[range(64), actions_to_index] = rewards + 0.95 * np.max(model.call(next_states).numpy(), axis=1) #############################################################

            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.square(target_y - model.call(states)))
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        m = np.mean(reward_mean)
        reward_mean_graph.append(m)
        last_x_position_graph.append(last_x_position)
        if last_x_position > 0.999 :
            model.save_weights('pendulum_g10.h5')
        print(f'episode : {i} reward_mean: {m:.3f} eps : {eps:.3f} last_x_position : {last_x_position:.3f}')


    plt.scatter(np.arange(0,episode), reward_mean_graph, label='reward_mean_scatter')
    reward_mean_graph = np.array(reward_mean_graph)
    rmgm = reward_mean_graph.reshape(int(episode/10),10)
    rmgm = np.mean(rmgm, axis=1)
    rmgm = rmgm.reshape(1,int(episode/10))[0]
    plt.plot(np.arange(0,episode,10), rmgm, color='red', linewidth=2.0, label='reward_mean')
    plt.legend(loc='lower right')
    plt.show()

    last_x_postion_mean_graph = np.array(last_x_position_graph)
    last_x_position_mean_graph = last_x_postion_mean_graph.reshape(int(episode/10),10)
    last_x_position_mean_graph = np.mean(last_x_position_mean_graph, axis=1)
    last_x_position_mean_graph = last_x_position_mean_graph.reshape(1,int(episode/10))[0]
    plt.plot(np.arange(0,episode,10), last_x_position_mean_graph, color='green', linewidth=2.0, label='last_x_position')
    plt.legend(loc='lower right')
    plt.show()

env.close()
```

- 결과

### - try1
![try1_rewardmean](https://github.com/leejongwon1234/Reinforcement_learning/assets/123047859/74b959e7-0f36-4a3c-8c4b-46837a17a6ea)
![try1_xposition](https://github.com/leejongwon1234/Reinforcement_learning/assets/123047859/e6d341ee-97b4-4662-a357-b2c9c8314043)
### - try2
![try2_rewardmean](https://github.com/leejongwon1234/Reinforcement_learning/assets/123047859/4ecaba40-e0e3-4aee-a7e0-91588212854e)
![try2_xposition](https://github.com/leejongwon1234/Reinforcement_learning/assets/123047859/65750457-49a0-4c59-a995-48e2c93f4208)
### - try3
![try3_rewardmean](https://github.com/leejongwon1234/Reinforcement_learning/assets/123047859/b6a6ac7b-c3f8-49ed-8e6c-e4c36e3bf698)
![try3_xposition](https://github.com/leejongwon1234/Reinforcement_learning/assets/123047859/dbdb2fa0-b272-4955-b878-860429f3ebad)
### - try4
![try4_rewardmean](https://github.com/leejongwon1234/Reinforcement_learning/assets/123047859/8f031164-aee9-47f4-8354-2c87bcb29dc5)
![try4_xposition](https://github.com/leejongwon1234/Reinforcement_learning/assets/123047859/618e3e02-332b-4c02-b278-ee7738ec7fc7)
### - try5
![try5_rewardmean](https://github.com/leejongwon1234/Reinforcement_learning/assets/123047859/e7964bc1-63a2-4d65-8145-501aecd0434b)
![try5_xposition](https://github.com/leejongwon1234/Reinforcement_learning/assets/123047859/7c612571-951f-4e23-8da7-b55433abbc59)
### - try6
![try6_rewardmean](https://github.com/leejongwon1234/Reinforcement_learning/assets/123047859/75d6e0d5-84f6-470b-8500-1cd28c52b93c)
![try6_xposition](https://github.com/leejongwon1234/Reinforcement_learning/assets/123047859/8d38b597-939d-4a28-a155-ce48c8e09309)
try1,2,3,4,5에서 다양한 학습을 하다가, try6에서 수렴하는 경향을 보인다.

- 테스트
```python
import gym
import numpy as np
from collections import deque
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import tensorflow as tf

# 뉴럴 네트워크 모델 만들기
class DQN(Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.d1 = Dense(128, input_dim=3, activation='relu')
        self.d2 = Dense(64, activation='relu')
        self.d3 = Dense(32, activation='relu')
        self.d4 = Dense(5, activation='linear')

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        return x

# pendulum 환경 구성
env = gym.make('Pendulum-v1', g=10, render_mode='human') #g=5

model = DQN()
model(np.zeros((1, 3)))  # 모델 호출하여 변수 생성

# 저장된 모델 로드
model.load_weights('pendulum_g10.h5')

episode = 10  # 테스트할 에피소드 수
step = 200

for i in range(episode):
    state = env.reset()[0]
    state = state.reshape(1, 3)
    reward_mean = []
    last_x_position = 0

    for t in range(step):
        # 테스트 시에는 epsilon-greedy 방법을 사용하지 않습니다.
        # 최적의 액션을 선택하기 위해 argmax 함수를 사용합니다.
        action = np.argmax(model(state)) - 2

        # 다음 상태 및 보상
        next_state, reward = env.step((action,))[0:2]
        next_state = next_state.reshape(1, 3)
        if t == step - 1:
            last_x_position = next_state[0][0]
        reward_mean.append(reward)

        state = next_state

    print(f'Test Episode: {i} Avg. Reward: {np.mean(reward_mean):.3f} Last x position: {last_x_position:.3f}')

env.close()
```
- 테스트 결과

![g10_test_out](https://github.com/leejongwon1234/Reinforcement_learning/assets/123047859/03152c1f-a8c6-4005-ab11-99e7e8065349)

![g=10video](https://github.com/leejongwon1234/Reinforcement_learning/assets/123047859/88097ead-8140-4418-b19f-092de200d55b)

<pre>
</pre>

## 5.시행착오 및 해결 방법, 피드백
<pre>
</pre>
### 1. 보상을 어떻게 처리 할 것인가?
<pre>
</pre>
>>이유는 알 수 없지만, reward를 임의로 조정하는 것 보다 환경에서 주어진 reward를 사용하는 것이 더 학습을 잘했다.

 - reward가 0.01보다 클때 reward+=1을 해주는 구조(0.01보다 큰 부분에서 불연속)로 변경해보았지만, 잘 학습하지 못했다.
 
 - np.square(reward)*(-1)을 이용하여 reward의 절댓값이 클수록 더 큰 penalty를 주는 구조로 변경한 경우 학습을 잘 했지만, 수렴하는 데에 더 많은 episode를 필요로 했고, reward 구조를 변경할 이유가 없었다.

 - np.sqrt(reward)*(-2)를 이용하여 reward의 절댓값이 작을수록 급격하게 reward가 커지는 방향으로 변경하는 경우 학습을 잘 했지만, 수렴하는 데에 더 많은 episode를 필요로 했고, reward구조를 변경할 이유가 없었다.
<pre>
</pre>
### 2.가중치를 언제 저장할 것인가?
<pre>
</pre>
>>last_x_postion이 0.999보다 큰 경우 성공이라고 생각하여 이 때의 가중치를 저장하고, 업데이트 하는 방향으로 구현하여 overfitting이나 원인불명의 학습실패를 방지하려 했지만, 결과론적으로 수렴이 되기 때문에 학습이 마무리 된 뒤, 가중치를 저장해도 문제가 없었을 것이다.
<pre>
</pre>
### 3.저장한 가중치를 어떻게 로드 하여 테스트 하는가?
<pre>
</pre>
>>단순히 모델을 구현하여 가중치를 로드 하니 계속 오류가 발생하여 고생했다. 이리저리 찾아보다가 모델 전부를 구현하고, 모델을 호출하여 변수를 생성한 뒤에 로드를 해야 문제없이 로드가 되었다.
```python
model = DQN()
model(np.zeros((1, 3)))  ### 모델 호출하여 변수 생성

# 저장된 모델 로드
model.load_weights('pendulum_g10.h5')
```
<pre>
</pre>
### 4.적절한 학습을 위해 코드를 어떤 구조로 구성해야할까?
<pre>
</pre>
>>g=1인 경우, 스윙이 필요없이 한쪽방향으로 힘을 가하여 막대를 세울 수 있기에 큰 고민이 필요없이 성공할 수 있지만, g가 5보다 커지는 경우 특정 상황에서 스윙이 필요하기 때문에 단순히 epsilon값이 줄어드는 방향으로 episode만 늘려서는 학습이 잘 되지 않았다. 

>>이를 위해 episode500인 학습을 3번 시도하는 방향으로 구조를 작성했고, 시도가 바뀔 때 마다 epsilon값을 reset(약간의 변형은 존재)하여 최적의 결과를 탐험하게 시도시켜 성공시켰다.
<pre>
</pre>



