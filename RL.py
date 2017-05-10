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

##


