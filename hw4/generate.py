import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as ly
import scipy.misc

import sys
testing_text_path = sys.argv[1]
import h5py
import json
import pickle as pkl

import copy
import math
from collections import OrderedDict, defaultdict
import random

import re
import os
from os.path import join, isfile
import funtion

#=========================
#Global parameters
#=========================
#一次要生成幾張圖
batch_size = 5
image_size = 64
# 雜訊的維度
z_dim = 100
t_dim =256
y_dim = 4800
gf_dim = 64
df_dim = 64
gfc_dim = 1024
channels = 3
caption_vector_length = 2400
data_format = 'NCHW'
device = '/gpu:0'
#建成dict一次丟進去
param_dict = {}
param_dict['batch_size'] = batch_size
param_dict['image_size'] = image_size
param_dict['z_dim'] = z_dim
param_dict['t_dim'] = t_dim
param_dict['y_dim'] = y_dim
param_dict['gf_dim'] = gf_dim
param_dict['df_dim'] = df_dim
param_dict['gfc_dim'] = gfc_dim
param_dict['channels'] = channels
param_dict['caption_vector_length'] = caption_vector_length
###############################################
#建立wgan的model
wgan = funtion.wgan(param_dict)
input_tensors, variables, loss, outputs, checks = wgan.build_model()

#load model
sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess,'./model')

#重建model
input_tensors, outputs = wgan.build_generator()

#讀取文字，並做前處理
testing_txtid_list= []
with open(testing_text_path,'r') as f:
	for line in f:
		line = line.split(',')[0]
		testing_txtid_list.append(line)	
with open(testing_text_path,'r') as f:
	row_text = f.read().split('\n')
	text_list = []
	text_h_e_list = []
	text_process_list = []
	for text in row_text:
		text_list.append(re.sub('\d+,', '', text))
	for text in text_list:
		text_h_e_list.append(re.sub('(hair|eyes)(\s+)', '\g<1> and ', text))
	for text in text_h_e_list:
		if len(text)>0:
			text_process_list.append(text)

#用skipthoughts轉成vector
skipthoughts_vectors = funtion.encode(funtion.load_model(),text_process_list)
#生成噪音
#z_noise = np.random.uniform(-1, 1,[batch_size,z_dim])

#依據文字生成圖片
word_image_dic = {}

for index,word_vector in enumerate(skipthoughts_vectors):
	word_image_list = []
	pkl_path = 'z_noise'+str(index)+'.pkl'

	#生成噪音
	z_noise = np.random.uniform(-1, 1, [batch_size,z_dim])
	#存噪音，之後才能重現
	with open (pkl_path,'wb') as f:
		pkl.dump(z_noise,f)
	#load z_noise
	#z_noise = pkl.load(open(pkl_path, 'rb'))
	#生成圖！！！！！
	text_batch_list = [word_vector[0:caption_vector_length]]*batch_size
	feed_dict =  {input_tensors['t_real_caption'] : text_batch_list,input_tensors['t_z'] : z_noise,}
	[wgan_image] = sess.run([outputs['generator']], feed_dict=feed_dict)
	#存起結果
	for i in range(0, batch_size):
		word_image_list.append(wgan_image[i,:,:,:])
	word_image_dic[index] = word_image_list
	
#放在sample資料夾下，檢查sample是否存在
"""
for f in os.listdir(join('samples')):
	if os.path.isfile(f):
		os.unlink(join('samples/' + f))
"""
path = os.getcwd()+'/samples/'
if not os.path.isdir(path):
	os.mkdir(path)
#輸出結果:一次輸出五張
for index1, index2 in enumerate(range(0,len(skipthoughts_vectors))):
		textid = testing_txtid_list[index1]
		for idx, img in enumerate(word_image_dic[index2]):
			scipy.misc.imsave(join('samples/sample_{}_{}.jpg'.format(textid,idx+1)),img)
print("ok!!!!!!")		

