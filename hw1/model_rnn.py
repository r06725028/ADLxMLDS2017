from collections import OrderedDict
import os
import re
import numpy as np
#from numpy import asarray, zeros
#import pandas as pd
#from keras.preprocessing.text import one_hot
#from keras.preprocessing.sequence import pad_sequences

#import tensorflow as tf
import keras
from keras import initializers, backend
from keras.utils import np_utils, to_categorical
from sklearn.cross_validation import train_test_split

from keras.layers.core import  Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, LSTM, TimeDistributed, Bidirectional
from keras.layers import Masking, Activation, Input,Conv1D, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
#from keras.callbacks import ModelCheckpoint

from keras.optimizers import Adam#不固定的梯度下降
import pickle

################################輸入argv############################### 
#import sys
#fbank_train_PATH = sys.argv[1]
#result_h5_PATH = sys.argv[2]

#####################################模型參數##########################
nb_epoch=10#訓練遍歷幾次
frame_samples=1124823#總frame數
seq_samples=3696#總句子數(最長frame數777)
#np.random.seed(1337)# for reproducibility
keras.initializers.Orthogonal(gain=1.0, seed=None)#初始化隨機變數

TIME_STEPS = 1# height:總共需讀幾次
#3696*0.85=3141=3*3*349=3*1047
INPUT_SIZE = 69#fbank維度
PADDING_SIZE = 31
BATCH_SIZE = 500#一次看500個
BATCH_INDEX = 0#用來生成數據
OUTPUT_SIZE = 39#每個frame的預測
CELL_SIZE = 512#hidden layer 要放多少cell
LR = 0.01

path = os.getcwd()#取得當前路徑

##############1.自建dict做binary轉換#######################################
inttobinary = {}
inttobinary['0']  = [1]+[0]*47
inttobinary['1']  = [0]+[1]+[0]*46
inttobinary['2']  = [0]*2+[1]+[0]*45
inttobinary['3']  = [0]*3+[1]+[0]*44
inttobinary['4']  = [0]*4+[1]+[0]*43
inttobinary['5']  = [0]*5+[1]+[0]*42
inttobinary['6']  = [0]*6+[1]+[0]*41
inttobinary['7']  = [0]*7+[1]+[0]*40
inttobinary['8']  = [0]*8+[1]+[0]*39
inttobinary['9']  = [0]*9+[1]+[0]*38 
inttobinary['10'] = [0]*10+[1]+[0]*37
inttobinary['11'] = [0]*11+[1]+[0]*36
inttobinary['12'] = [0]*12+[1]+[0]*35
inttobinary['13'] = [0]*13+[1]+[0]*34
inttobinary['14'] = [0]*14+[1]+[0]*33
inttobinary['15'] = [0]*15+[1]+[0]*32
inttobinary['16'] = [0]*16+[1]+[0]*31
inttobinary['17'] = [0]*17+[1]+[0]*30
inttobinary['18'] = [0]*18+[1]+[0]*29
inttobinary['19'] = [0]*19+[1]+[0]*28
inttobinary['20'] = [0]*20+[1]+[0]*27
inttobinary['21'] = [0]*21+[1]+[0]*26
inttobinary['22'] = [0]*22+[1]+[0]*25
inttobinary['23'] = [0]*23+[1]+[0]*24 
inttobinary['24'] = [0]*24+[1]+[0]*23
inttobinary['25'] = [0]*25+[1]+[0]*22
inttobinary['26'] = [0]*26+[1]+[0]*21
inttobinary['27'] = [0]*27+[1]+[0]*20
inttobinary['28'] = [0]*28+[1]+[0]*19
inttobinary['29'] = [0]*29+[1]+[0]*18
inttobinary['30'] = [0]*30+[1]+[0]*17
inttobinary['31'] = [0]*31+[1]+[0]*16
inttobinary['32'] = [0]*32+[1]+[0]*15
inttobinary['33'] = [0]*33+[1]+[0]*14
inttobinary['34'] = [0]*34+[1]+[0]*13
inttobinary['35'] = [0]*35+[1]+[0]*12
inttobinary['36'] = [0]*36+[1]+[0]*11
inttobinary['37'] = [0]*37+[1]+[0]*10
inttobinary['38'] = [0]*38+[1]+[0]*9
inttobinary['39'] = [0]*39+[1]+[0]*8
inttobinary['40'] = [0]*40+[1]+[0]*7
inttobinary['41'] = [0]*41+[1]+[0]*6
inttobinary['42'] = [0]*42+[1]+[0]*5
inttobinary['43'] = [0]*43+[1]+[0]*4
inttobinary['44'] = [0]*44+[1]+[0]*3
inttobinary['45'] = [0]*45+[1]+[0]*2
inttobinary['46'] = [0]*46+[1]+[0]
inttobinary['47'] = [0]*47+[1]

###############2.自建dict做字母到數字的轉換##############
"""
chartoint = {}#用chartoint查表做數字轉換
f = open("./data/48phone_char.map","r")
for line in f:#48個對應
    #numlist = re.split('[\s]',line)#用空白分#print("numlist ",numlist)#['aa',' ','0']
    #print(numlist[0]+"|"+numlist[1])
    (ch1, integer, ch2) = line.split('\t')
    #chartoint[numlist[0]] = inttobinary[numlist[1]]#一層list
    chartoint[ch1] = inttobinary[integer]
f.close()
"""
chartoint = {}#用chartoint查表做數字轉換
chartoint['aa'] = inttobinary['0']
chartoint['ae'] = inttobinary['1']
chartoint['ah'] = inttobinary['2']
chartoint['ao'] = inttobinary['3']
chartoint['aw'] = inttobinary['4']
chartoint['ax'] = inttobinary['5']
chartoint['ay'] = inttobinary['6']
chartoint['b'] = inttobinary['7']
chartoint['ch'] = inttobinary['8']
chartoint['cl'] = inttobinary['9']
chartoint['d'] = inttobinary['10']
chartoint['dh'] = inttobinary['11']
chartoint['dx'] = inttobinary['12']
chartoint['eh'] = inttobinary['13']
chartoint['el'] = inttobinary['14']
chartoint['en'] = inttobinary['15']
chartoint['epi'] = inttobinary['16']
chartoint['er'] = inttobinary['17']
chartoint['ey'] = inttobinary['18']
chartoint['f'] = inttobinary['19']
chartoint['g'] = inttobinary['20']
chartoint['hh'] = inttobinary['21']
chartoint['ih'] = inttobinary['22']
chartoint['ix'] = inttobinary['23']
chartoint['iy'] = inttobinary['24']
chartoint['jh'] = inttobinary['25']
chartoint['k'] = inttobinary['26']
chartoint['l'] = inttobinary['27']
chartoint['m'] = inttobinary['28']
chartoint['n'] = inttobinary['29']
chartoint['ng'] = inttobinary['30']
chartoint['ow'] = inttobinary['31']
chartoint['oy'] = inttobinary['32']
chartoint['p'] = inttobinary['33']
chartoint['r'] = inttobinary['34']
chartoint['s'] = inttobinary['35']
chartoint['sh'] = inttobinary['36']
chartoint['sil'] = inttobinary['37']
chartoint['t'] = inttobinary['38']
chartoint['th'] = inttobinary['39']
chartoint['uh'] = inttobinary['40']
chartoint['uw'] = inttobinary['41']
chartoint['v'] = inttobinary['42']
chartoint['vcl'] = inttobinary['43']
chartoint['w'] = inttobinary['44']
chartoint['y'] = inttobinary['45']
chartoint['z'] = inttobinary['46']
chartoint['zh'] = inttobinary['47']

################################3.資料前處理#####################################
train_x = []
train_y = []

##1.讀fbank
data = OrderedDict()#用dict存維度資料
f = open('./data/fbank/train.ark',"r+")
for line in f:#1124823個訊框
    numlist = line.split()#用空白分
    allid = numlist[0]#id部分
    values = numlist[1:]#維度部分
    (spid, senid, frid) = allid.split('_')#切出句子的id
    seqID = spid+'_'+senid
    if  seqID in data.keys():#同一句相加
    	data[seqID].append(np.asarray(values,'f'))#轉為array放入
    else:#沒有則新建
    	data[seqID] = [np.asarray(values,'f')]
f.close()

#print (data.keys())

##2.讀label#已check
f = open(path+'/sorted_train_label.lab',"r")
for line in f:
	(allid, lab) = line.split(',')
	lab = lab.strip('\n')
	train_y.append(chartoint[lab])#兩層list
f.close()

print("read data ok")

#########################################做padding###########################################
for seqID in data.keys():#對每一句#已check
	for index in range(len(data[seqID])):#每個frame
		padding = []#抓前後來補
		
		begin = -int((PADDING_SIZE-1)/2)#開頭
		end = int((PADDING_SIZE+1)/2)#結尾
		
		for i in range(begin, end):
			if (index + i < 0 or index + i > len(data[seqID])-1 ):
				pad = [0.0] * 69#不夠就補零
				padding.append(pad)
			else:#夠就直接放入
				padding.append(data[seqID][index+i])

		train_x.append(padding)

input_shape = np.array(train_x).shape
print(input_shape)
print ("padding ok")

##############################################建立RNN的模型###############################
model = Sequential()
#做dropout??
#model.add(Conv1D(128, padding = 'causal', kernel_size = 10))
#model.add(SimpleRNN(128,stateful = False,return_sequences=True))
model.add(BatchNormalization(input_shape=(input_shape[1], input_shape[2])))

model.add(LSTM(512, return_sequences=False))
#model.add(Activation('relu'))

model.add(Dense(512, activation='relu'))
#model.add(Activation('tanh'))
model.add(Dense(512, activation='relu'))

model.add(Dense(48, activation='softmax'))
#model.add(Activation('softmax'))
print ("build model ok")
################################################訓練模型#############################################
model.compile(loss='categorical_crossentropy',optimizer='adam' , metrics=['accuracy'])
print(model.summary())

model.fit(train_x, train_y, epochs=10, verbose=1, validation_split=0.1, batch_size=500)

print ("train ok")

###############################################存起model#############################################
model.save('my_model_2.h5')
#model.save(result_h5_PATH)

backend.clear_session()

print ("save model ok")
