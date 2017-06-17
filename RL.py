##Reinforcement Learning

##1 2 3 RL
#nothing, unsupervise AlphaGO, teacher only give score
#action>score

#表格:Qlearning, Sarsa, DNN: Deep Qnetwork
#Behavior base:Policy gradient
#Environment: Model-based RL 

#Model-free RL不理解 vs Model-based RL理解建立模型
#qlearning, sarsa, policy gd is model-free, but could apply to model-based
#model-base has imagination, not necs real

#Policy-based RL雞綠 vs Value-based RL價值
#prob based: anything could be possible by prob, could continue
	#policy gd
#value based: only high value, only discrete
	#qlearning, sarsa
#combine as actor-critic

#monte-carlo update回合 vs temporal-difference update單步
#mc: till all end then update, basic po gd, mc learning
#td: update each step, qlearning, sarsa, adv po gd, more effi

#on-policy vs off-polich
#on: self learning, sarsa, sarsa lambda
#off: by looking or memory, qlearning, deep q network

#earn high score

##4
#tkinter>>OpenAI gym

##5 Qlearning
#action based, decision
#hw>tv>tv>penal
#hw>hw>hw>prize

#table when at s1 stage,
#   a1-tv  a2-hw
#s1 -2     1 
#then choose a2 for higher value

#table when at s2 stage,
#   a1-tv  a2-hw
#s2 -4     2
#then choose a2 for also higher value

#repeat....

#Qtable: 
#epsilon-greedy if =0.9 表示九成都罩著Q表
#alpha 學習笑率 更新Q表的速度 <1
#gamma 衰退值 wieght眼光在未來的獎勵或最近的獎勵 (0,1)
#Q(s,a) := Q(s,a)+alpha[R+gamma max_a' Q(s',a')-Q(s,a)]



##6 example
import numpy as np
import pandas as pd
import time

np.random.seed(2)  # reproducible


N_STATES = 6   # the length of the 1 dimensional world
ACTIONS = ['left', 'right']     # available actions
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 13   # maximum episodes
FRESH_TIME = 0.1    # fresh time for one move


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
    # print(table)    # show table
    return table


def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):  
    # 90%prob follow Qtable act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = state_actions.argmax()
    return action_name


def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    if A == 'right':    # move right
        if S == N_STATES - 2:   # terminate
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:   # move left
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, R


def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter) #更新環境
        while not is_terminated:

            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)  # take action & get next state and reward
            q_predict = q_table.ix[S, A]
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal
            else:
                q_target = R     # next state is terminal
                is_terminated = True    # terminate this episode

            q_table.ix[S, A] += ALPHA * (q_target - q_predict)  # update
            S = S_  # move to next state

            update_env(S, episode, step_counter+1) #更新環境
            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
	print(q_table)


