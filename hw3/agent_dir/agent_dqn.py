from agent_dir.agent import Agent
import tensorflow as tf 
import numpy as np 
import random
tf.set_random_seed(1)
from collections import deque 
#import os
import cv2
#import pandas as pd
#import pickle

def lrelu(x, alpha):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


#===============================================================
# Hyper Parameters:
#FRAME_PER_ACTION = 1#不用了
GAMMA = 0.99#0.95# decay rate of past observations
OBSERVE = 10000.#50000. # timesteps to observe before training
EXPLORE = 1000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.05#0.1#0.001 # final value of epsilon
INITIAL_EPSILON = 1.0#0.01 # starting value of epsilon
REPLAY_MEMORY = 10000#1000000 # number of previous transitions to remember
BATCH_SIZE = 32 # size of minibatch
UPDATE_TIME = 1000#10000
ENVIRONMENT_STEPS = 1e7
FRAMES = 4
#==================================================================

class Agent_DQN(Agent):
  def __init__(self, env, args): 
    super(Agent_DQN,self).__init__(env)

    #if args.test_dqn:
      #you can load your model here
      #print('loading trained model')
    ####################################my###########################
    # init environment
    self.env = env
    # init replay memory
    self.replayMemory = deque()
    # init some parameters
    self.timeStep = 0
    self.epsilon = INITIAL_EPSILON
    self.actions = 4
    # init Q network
    self.stateInput,self.QValue,self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,\
      self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2 = self.createQNetwork()
    # init Target Q Network
    self.stateInputT,self.QValueT,self.W_conv1T,self.b_conv1T,self.W_conv2T,self.b_conv2T,self.W_conv3T,\
      self.b_conv3T,self.W_fc1T,self.b_fc1T,self.W_fc2T,self.b_fc2T = self.createQNetwork()
    # a sequence of tf operation
    self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1), self.b_conv1T.assign(self.b_conv1),\
      self.W_conv2T.assign(self.W_conv2), self.b_conv2T.assign(self.b_conv2), self.W_conv3T.assign(self.W_conv3),\
      self.b_conv3T.assign(self.b_conv3), self.W_fc1T.assign(self.W_fc1), self.b_fc1T.assign(self.b_fc1),\
      self.W_fc2T.assign(self.W_fc2), self.b_fc2T.assign(self.b_fc2)]

    self.createTrainingMethod()

    # saving and loading networks
    self.saver = tf.train.Saver()
    self.session = tf.InteractiveSession()
    self.session.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state("mydqn_ckpt")
    #if checkpoint and checkpoint.model_checkpoint_path:
    #self.saver.restore(self.session, checkpoint.model_checkpoint_path)
    path = 'my_dqn_model'
    self.saver.restore(self.session,path)
    #print("Successfully loaded:", checkpoint.model_checkpoint_path)
    print("Successfully loaded:", path)
    #else:
        #print("Could not find old network weights")
    ###########################################################################
  def init_game_setting(self):
    #初始為零
    self.epsilon = 0.0

  def train(self):
    ###################################################
    #1.初始值為零
    episode = 0
    sum_reward = 0
    avg_reward_list = []#算平均用的list 
    #2.加上random#用11037#1
    self.env.seed(11037)
    #3.reset現在的狀態
    self.currentState = self.env.reset()
    #4.開始train
    for _ in range(int(ENVIRONMENT_STEPS)):
      #根據現在狀態去選擇action
      action = self.make_action(self.currentState)
      #更新time_step資訊
      nextObservation, reward, terminal, info = self.env.step(action)
      #初始化每個動作的機率皆為零
      action_list = [0.0] * 4
      #把選擇的那個action的機率設為1，其他仍為零
      action_list[action] = 1.0
      #轉為np.array放入函式中
      action_list = np.array(action_list)
      self.setPerception(nextObservation, action_list, reward, terminal)
      #把reward加總起來
      sum_reward += reward
      #一局結束的話，輸出結果
      if terminal:
        #更新ep
        episode += 1
        #判斷長度，用來算100期的移動平均，超過一百就把最舊的刪掉，以加入最新一期的
        if(len(avg_reward_list) > 100): 
          del avg_reward_list[0]
        #加入最新一期的
        avg_reward_list.append(sum_reward)
        #印出結果
        print('Episode:'+str(episode),' sum_reward:'+'{0:.1f}'.format(sum_reward),\
          ' avg_reward:'+'{0:.2f}'.format(np.mean(avg_reward_list)))
        #歸零
        sum_reward = 0
        #1把結果寫入txt方便畫圖！！
        if(episode % 100 == 0):
          with open('dqn_graph.txt', 'a') as f:
            f.write('Episode:'+str(episode)+'\tavg_reward:'+'{0:.2f}'.format(np.mean(avg_reward_list))+\
              '\tTimeSteps:'+str(self.timeStep)+'\tEpsilon:'+str(self.epsilon)+'\n')
        #重新reset
        self.currentState = self.env.reset()
        #####################################################
  def make_action(self, observation, test=True):
    #########################################################
    #原本的getAction函數
    QValue = self.QValue.eval(feed_dict= { self.stateInput: [observation] })[0]
    #action = np.zeros(self.actions)
    #action_index = 0
    action = 0
    #if self.timeStep % FRAME_PER_ACTION == 0:
    if random.random() <= self.epsilon:
      action = random.randrange(self.actions)
      #action[action_index] = 1
    else:
      action = np.argmax(QValue)
      #action[action_index] = 1
    #else:
      #action[0] = 1 # do nothing

    # change episilon
    if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
      self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

    return action
#######################################另外定義的函數###################################
  def createTrainingMethod(self):
    self.actionInput = tf.placeholder("float", [None, self.actions])
    self.yInput = tf.placeholder("float", [None]) 
    Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), axis=1)
    self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
    self.trainStep = tf.train.RMSPropOptimizer(1e-4, 0.99).minimize(self.cost)#decay改成0.99!!!!!

  def setPerception(self, newState, action, reward, terminal):
    #newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
    self.replayMemory.append((self.currentState, action, reward, newState, terminal))
    if len(self.replayMemory) > REPLAY_MEMORY:
      self.replayMemory.popleft()
    if self.timeStep > OBSERVE: 
       # Train the network
      self.trainQNetwork()
    """
    # print info
    state = ""
    if self.timeStep <= OBSERVE:
      state = "observe"
    elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
      state = "explore"
    else:
      state = "train"

    print "TIMESTEP", self.timeStep, "/ STATE", state, \
    "/ EPSILON", self.epsilon
    """
    self.currentState = newState
    self.timeStep += 1

  def trainQNetwork(self):
    # Step 1: obtain random minibatch from replay memory
    minibatch = random.sample(self.replayMemory, BATCH_SIZE)
    state_batch = [data[0] for data in minibatch] 
    action_batch = [data[1] for data in minibatch] 
    reward_batch = [data[2] for data in minibatch] 
    nextState_batch = [data[3] for data in minibatch] 

    # Step 2: calculate y 
    y_batch = []
    QValue_batch = self.QValueT.eval(feed_dict={ self.stateInputT: nextState_batch })
    for i in range(0, BATCH_SIZE):
      terminal = minibatch[i][4]
      if terminal:
        y_batch.append(reward_batch[i])
      else:
        y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i])) 
    #一次看四個frame
    if self.timeStep % FRAMES == 0:
      self.trainStep.run(feed_dict={self.yInput : y_batch,self.actionInput : action_batch,self.stateInput : state_batch})
    # save network every 100000 iteration
    if self.timeStep % 10000 == 0:
      self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step = self.timeStep)

    if self.timeStep % UPDATE_TIME == 0:
      self.copyTargetQNetwork()

  def copyTargetQNetwork(self):
    self.session.run(self.copyTargetQNetworkOperation)

  def createQNetwork(self):
    # network weights
    W_conv1 = self.weight_variable([8, 8, 4, 32])
    b_conv1 = self.bias_variable([32])

    W_conv2 = self.weight_variable([4, 4, 32, 64])
    b_conv2 = self.bias_variable([64])

    W_conv3 = self.weight_variable([3, 3, 64, 64])
    b_conv3 = self.bias_variable([64])

    W_fc1 = self.weight_variable([3136, 512])
    b_fc1 = self.bias_variable([512])

    W_fc2 = self.weight_variable([512,self.actions])
    b_fc2 = self.bias_variable([self.actions])

    # input layer
    stateInput = tf.placeholder("float", [None, 84, 84, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(self.conv2d(stateInput, W_conv1, 4) + b_conv1)
    #h_pool1 = self.max_pool_2x2(h_conv1)
    h_conv2 = tf.nn.relu(self.conv2d(h_conv1, W_conv2, 2) + b_conv2)
    h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)
    # h_conv3_shape = h_conv3.get_shape().as_list()
    #print ("dimension:",h_conv3_shape[1]*h_conv3_shape[2]*h_conv3_shape[3])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 3136])
    #h_fc1 = tf.nn.leaky_relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1, alpha=0.01)
    #lrelu(x, alpha):
    h_fc1 = lrelu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1, alpha=0.01)

    # Q Value layer
    QValue = tf.matmul(h_fc1, W_fc2) + b_fc2

    return stateInput,QValue,W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2
 
  def weight_variable(self, shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

  def bias_variable(self, shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

  def conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="VALID")

  def max_pool_2x2(self,x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
