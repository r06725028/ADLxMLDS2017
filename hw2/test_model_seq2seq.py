#-*- coding: utf-8 -*-
#__author__ = "Xinpeng.Chen"

import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow.python.training import optimizer
from tensorflow.python.ops import variable_scope

#讀丟入的參數
import sys
data = sys.argv[1]
test_output_filename = sys.argv[2]
peer_review_output_filename = sys.argv[3]
#讀檔
import os
import json#讀label

#處理句子
#import ipdb
import time
#import cv2
from keras.preprocessing import sequence
#import matplotlib.pyplot as plt
from random import randint#隨機亂數---做attention
import string
from collections import OrderedDict #有序字典
#====================================================================================
#                                    包含各種function的class
#====================================================================================
class Video_Caption_Generator():
##############################一、初始化function(start)######################################
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, 
        n_video_lstm_step, n_caption_lstm_step, bias_init_vector=None):
        #################1.設定參數
        self.dim_image = dim_image#每個影片抽出的每張圖片的維度:4096
        self.n_words = n_words#總字彙數量:看wordtoix的數量
        self.dim_hidden = dim_hidden#nn cell數:256(1000)
        self.batch_size = batch_size#一批的個數:50(or 100??)
        self.n_lstm_steps = n_lstm_steps#這是甚麼(和n_video_lstm_step一樣??????????)
        self.n_video_lstm_step=n_video_lstm_step#一個影片切成幾張圖片:80
        self.n_caption_lstm_step=n_caption_lstm_step#句子的最高字數上限
        ##############2.隨機產生-0.1~0.1之間的數值，shape=[n_words, dim_hidden]和[dim_hidden, n_words]:初始化句子weight矩陣
        with tf.device("/cpu:0"):#指定cpu
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')#輸入hidden layer  
        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W')#從hidden layer輸出
        #############3.看是否已有bias_init_vector::初始化句子bias向量
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')#有的話拿來用
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')#沒有的話用全零代替 
        ###############3.設定lstm層的cell數/狀態
        self.lstm1 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)#encoding階段(預備)
        self.lstm2 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)#decoding階段(實際預測)
        ############4.隨機產生-0.1~0.1之間的數值，shape=[dim_image, dim_hidden]:初始化影片weight矩陣
        self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
        ############5.產生長度為[dim_hidden]的全零array#初始化影片bias的值     
        self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')
        ###########6.attenation(類似上面)
        self.weight_matrix = tf.Variable(tf.random_uniform([self.lstm2.state_size * 2, self.lstm2.state_size], -0.1,0.1), name='attention_W')
        self.bias_matrix = tf.Variable(tf.zeros([self.lstm2.state_size]), name='attention_b')
##############################一、初始化function(end)######################################
###############################三、生成句子的function(start)#################################
    def build_generator(self):#和bulid model的function一樣，只是改成讀影片feature生成句子
        ##########1.一個影片的feature輸入維度為(1,80,4096):一個影片生一個句子
        video = tf.placeholder(tf.float32, [1, self.n_video_lstm_step, self.dim_image])
        ##########2.做影片的feature的mask
        video_mask = tf.placeholder(tf.float32, [1, self.n_video_lstm_step])
        ##########3.reshape為一整列
        video_flat = tf.reshape(video, [-1, self.dim_image])
        ###########4.feature做input(x)乘上weight(w)再加上bias(b)
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        ##########5.再reshape回去為(1,80,256)
        image_emb = tf.reshape(image_emb, [1, self.n_video_lstm_step, self.dim_hidden])
        ##########6.產生長度不同的零array????????????做甚麼用
        state1 = tf.zeros([1, self.lstm1.state_size])
        state2 = tf.zeros([1, self.lstm2.state_size])
        padding = tf.zeros([1, self.dim_hidden])
        ##########7.存放句子的list
        generated_words = []
        ##########8.存放機率分布的list
        probs = []
        ##########9.存放每個字embedding的list
        embeds = []
############################三之一、Encoding Stage((同上同上同上))##############################
        #讀每個影片的所有圖片，依序生成句子
        for i in range(0, self.n_video_lstm_step):
            ##1.除了第一次以外，重複使用參數
            #if i > 0:#放入lstm1和2裡面
                #tf.get_variable_scope().reuse_variables()
            ###2.lstm1 layer
            with tf.variable_scope("LSTM1"):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                output1, state1 = self.lstm1(image_emb[:, i, :], state1)
            ###3. lstm2 layer
            with tf.variable_scope("LSTM2"):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                output2, state2 = self.lstm2(tf.concat( [padding, output1],1), state2)
            ###4.增加維度，調整shape(for att)
            expand_state2 = tf.expand_dims(state2, 2)
            if i == 0:
                new_state2 = expand_state2
            else:
                new_state2 = tf.concat([new_state2, expand_state2], 2)
########################三之二、Decoding Stage((同上同上同上))######################################
        for i in range(0, self.n_caption_lstm_step):## Phase 2 => only generate captions
            #tf.get_variable_scope().reuse_variables()#!!!!!!!!!!需刪掉
            #####1.查詢現在這個字的embedding    
            if i == 0:
                with tf.device('/cpu:0'):
                    tf.get_variable_scope().reuse_variables()#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@先刪這行
                    current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([1], dtype=tf.int64))
            ##2.重複使用參數??放到current_embed內??
            #tf.get_variable_scope().reuse_variables()@@@先刪這行
            ###3.lstm1 layer
            with tf.variable_scope("LSTM1"):
                tf.get_variable_scope().reuse_variables()
                output1, state1 = self.lstm1(padding, state1)
            ###4.lstm2 layer
            with tf.variable_scope("LSTM2"):
                tf.get_variable_scope().reuse_variables()
                ###5.增加維度，調整shape
                expand_state2 = tf.expand_dims(state2, 2)
                ##6.計算多維的相似性
                ##沿着tensor的某些维度求和
                sim = tf.reduce_sum(tf.multiply(new_state2, expand_state2), 1, keep_dims=True)
                ##用softmax作正規化
                softmax_sim = tf.nn.softmax(sim)
                ##矩陣相乘
                matmul_sim = tf.matmul(new_state2, tf.transpose(softmax_sim, perm=[0,2,1]))
                ##返回把長度為1的維度去掉的ndarray(少一維)
                squeeze_sim = tf.squeeze(matmul_sim, [2])
                ##在xw_plus_b後再用tanh做activeation function
                state2_tanh = tf.tanh(tf.matmul(tf.concat([state2, squeeze_sim], 1), self.weight_matrix) + self.bias_matrix)
                ##把output和current_embed結合起來
                output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2_tanh)
                output2, state2 = self.lstm2(tf.concat( [current_embed, output1],1), state2)
            ##7.計算權重，矩陣相乘再加上bias
            logit_words = tf.nn.xw_plus_b( output2, self.embed_word_W, self.embed_word_b)
            #從onehot vector中取出最高機率的index
            max_prob_index = tf.argmax(logit_words, 1)[0]
            #放入生成句子的list中
            generated_words.append(max_prob_index)
            #把計算出的權重放入
            probs.append(logit_words)
            with tf.device("/cpu:0"):
                #查詢最高機率的index對應到的word embedding
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                #擴展維度(調整shape?????)
                current_embed = tf.expand_dims(current_embed, 0)
            #放入list中
            embeds.append(current_embed)
        return video, video_mask, generated_words, probs, embeds
###############################三、生成句子的function(end)#################################
#=====================================================================================
# Global Parameters
#=====================================================================================
video_path = data+'/testing_data/video'

video_p_feat_path = data+'/peer_review/feat'
video_test_feat_path = data+'/testing_data/feat'

video_train_data_path = data+'/training_label.json'
video_test_data_path = data+'/testing_id.txt'

model_path = './models'
#=======================================================================================
# Train Parameters
#=======================================================================================
#train feature = (1450,80,4096)
#助教建議的參數
dim_image = 4096
dim_hidden= 256

n_video_lstm_step = 80
n_caption_lstm_step = 25#生成句子的最大長度
n_frame_step = 80#這是甚麼(和n_video_lstm_step一樣??????????)

n_epochs = 200
batch_size = 200
learning_rate = 0.001
#========================================================================================
#                                    讀檔
#========================================================================================
###############################一、讀csv檔#############################################
#讀test檔的feature
def get_video_test_data(video_data_path, video_feat_path):
    print ("test讀檔read start")#用原作者的才行
    #1.用csv讀
    video_data = pd.read_csv(video_data_path, header=None)
    #2.新增欄位'video_path'
    #video_data['video_path'] = video_data.apply(lambda row: row['id'] +'.npy', axis=1)
    video_data['video_path'] = video_data.apply(lambda row: row[0] +'.npy', axis=1)
    #3.加上video_train_feat_path路徑((feature!!!!))，結合label和feature
    video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(video_feat_path, x))
    #4.檢查檔案是否存在(感覺不需要)
    video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists( x ))]
    #5.檢查是不是str:是list!!!
    # video_data = video_data[video_data['Description'].map(lambda x: isinstance(x, str))]
    #6.排序所有avi檔案名#????????????????感覺不需要?????????????????
    unique_filenames = sorted(video_data['video_path'].unique())
    #7.移除重覆、重新排序
    test_data = video_data[video_data['video_path'].map(lambda x: x in unique_filenames)]
    print ("讀檔read ok!!!")
    return test_data

###############################二、讀txt檔#############################################
#讀peer檔的feature
print ("peer讀檔read start")
filename = data+'/peer_review_id.txt ' #输入要遍历读取的文件路径及文件名
file = open(filename,'r')
done = 0
#dict = OrderedDict()
peer_list = []
while not  done:
        aLine = file.readline()
        if(aLine != ''):
            #print aLine
            aLine = aLine.strip()+'.npy'
            peer_list.append(aLine)
        else:
            done = 1
file.close()   #关闭文件
print ("peer讀檔read ok!!!")
  

#========================================================================================
#                                    開始測試!!
#========================================================================================
def test(model_path='./your_seq2seq_model'):###############改路徑!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #1.取得feature
    test_data = get_video_test_data(video_test_data_path, video_test_feat_path)
    #2.移除重複
    test_videos = test_data['video_path'].unique()
    #3.下載ixtoword和bias_init_vector
    ixtoword = pd.Series(np.load('./ixtoword.npy').tolist())
    bias_init_vector = np.load('./bias_init_vector.npy')
    #4.初始化模型
    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            n_video_lstm_step=n_video_lstm_step,
            n_caption_lstm_step=n_caption_lstm_step,
            bias_init_vector=bias_init_vector)
    #5.建立模型
    video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()
    #6.使用互動式sess(會自動預設)
    sess = tf.InteractiveSession()
    #7.儲存模型
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    ##############################8.開啟預測的txt檔###############################
    test_output_txt_fd = open(test_output_filename, 'w')
    for idx, video_feat_path in enumerate(test_videos):
        #9.印出video的id
        print (idx, video_feat_path)
        #10.從路徑去開啟feature檔讀取
        video_feat = np.load(video_feat_path)[None,...]
        #video_feat = np.load(video_feat_path)
        #video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
        #11.做mask
        if video_feat.shape[1] == n_frame_step:
            video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
        else:#不須做
            continue
            #shape_templete = np.zeros(shape=(1, n_frame_step, 4096), dtype=float )
            #shape_templete[:video_feat.shape[0], :video_feat.shape[1], :video_feat.shape[2]] = video_feat
            #video_feat = shape_templete
            #video_mask = np.ones((video_feat.shape[0], n_frame_step))
        #12.生成句子
        generated_word_index, probs = sess.run([caption_tf, probs_tf], feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
        #13.用ixtoword作轉換
        generated_words = ixtoword[generated_word_index]
        #14.用argmax取出最大的句子長度
        punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
        #15.以最長句子做標準取預測的字數
        generated_words = generated_words[:punctuation]
        #16.把每次預測的字串連接起來
        generated_sentence = ' '.join(generated_words)
        #17.去掉我們自己加上去的bos和eos
        generated_sentence = generated_sentence.replace('<bos> ', '')
        generated_sentence = generated_sentence.replace(' <eos>', '')
        #18.印出預測的句子
        print(generated_sentence,'\n')
        #19.寫入txt檔中(注意格式!!!!!!!!!!!!!!)
        test_output_txt_fd.write(video_feat_path.replace('./MLDS_hw2_data/testing_data/feat/', '').replace('.npy', ''))#只要取id部分
        test_output_txt_fd.write(',' + generated_sentence + '\n')
        #p

#========================================================================================
#                                    開始peer review!!
#========================================================================================
def p_test(model_path='./your_seq2seq_model'):###############改路徑!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #1.取得feature
    #test_data = get_video_test_data(video_test_data_path, video_test_feat_path)
    #2.移除重複
    #test_videos = test_data['video_path'].unique()
    test_videos = peer_list
    #3.下載ixtoword和bias_init_vector
    ixtoword = pd.Series(np.load('./ixtoword.npy').tolist())
    bias_init_vector = np.load('./bias_init_vector.npy')
    #4.初始化模型
    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            n_video_lstm_step=n_video_lstm_step,
            n_caption_lstm_step=n_caption_lstm_step,
            bias_init_vector=bias_init_vector)
    #5.建立模型
    video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()
    #6.使用互動式sess(會自動預設)
    sess = tf.InteractiveSession()
    #7.儲存模型
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    ##############################8.開啟預測的txt檔###############################
    test_output_txt_fd = open(peer_review_output_filename, 'w')
    for idx, video_feat_path in enumerate(test_videos):
        #9.印出video的id
        print (idx, video_feat_path)
        #10.從路徑去開啟feature檔讀取
        video_feat = np.load(video_feat_path)[None,...]
        #video_feat = np.load(video_feat_path)
        #video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
        #11.做mask
        if video_feat.shape[1] == n_frame_step:
            video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
        else:#不須做
            continue
            #shape_templete = np.zeros(shape=(1, n_frame_step, 4096), dtype=float )
            #shape_templete[:video_feat.shape[0], :video_feat.shape[1], :video_feat.shape[2]] = video_feat
            #video_feat = shape_templete
            #video_mask = np.ones((video_feat.shape[0], n_frame_step))
        #12.生成句子
        generated_word_index, probs = sess.run([caption_tf, probs_tf], feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
        #13.用ixtoword作轉換
        generated_words = ixtoword[generated_word_index]
        #14.用argmax取出最大的句子長度
        punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
        #15.以最長句子做標準取預測的字數
        generated_words = generated_words[:punctuation]
        #16.把每次預測的字串連接起來
        generated_sentence = ' '.join(generated_words)
        #17.去掉我們自己加上去的bos和eos
        generated_sentence = generated_sentence.replace('<bos> ', '')
        generated_sentence = generated_sentence.replace(' <eos>', '')
        #18.印出預測的句子
        print(generated_sentence,'\n')
        #19.寫入txt檔中(注意格式!!!!!!!!!!!!!!)
        test_output_txt_fd.write(video_feat_path.replace('./MLDS_hw2_data/testing_data/feat/', '').replace('.npy', ''))#只要取id部分
        test_output_txt_fd.write(',' + generated_sentence + '\n')
        #p
########實際測試###########        
print("test begin")
test()
print("test ok")

print("peer begin")
p_test()
print("peer ok")

##############peer_review