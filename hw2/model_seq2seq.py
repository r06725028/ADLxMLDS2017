#-*- coding: utf-8 -*-
#__author__ = "Xinpeng.Chen"

import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow.python.training import optimizer
from tensorflow.python.ops import variable_scope

#讀丟入的參數
import sys

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
##############################一、初始化function(end)######################################

##############################二、建立模型function(start)######################################
    def build_model(self):#讀影片feature的model(build_generator一樣!!)
        ##########1.影片的feature輸入維度為(batch_size,80,4096)
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step, self.dim_image])
        ##########2.做影片的feature的mask
        video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step])
        ##########3.reshape為一整列        
        video_flat = tf.reshape(video, [-1, self.dim_image])
        ###########4.feature做input(x)乘上weight(w)再加上bias(b)
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b )#(batch_size*n_lstm_steps, dim_hidden)
        ##########5.再reshape回去為(batch_size,80,256)     
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_steps, self.dim_hidden])
        ##########6.caption輸入維度為(batch_size,80,n_caption_lstm_step+1):加一維代表padding的部分!!
        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_caption_lstm_step+1])
        ##########7.做caption的mask
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_caption_lstm_step+1])
        ##########8.產生長度不同的零array????????????做甚麼用
        state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
        padding = tf.zeros([self.batch_size, self.dim_hidden])    
        #########9.輸出機率分布(list)
        probs = []
        #########10,初始化loss為零
        loss = 0.0
#################################二之一、Encoding Stage ##################################
        for i in range(0, self.n_video_lstm_step): ## Phase 1 => 讀影片feature&預備generate captions
            ##1.除了第一次以外，重複使用參數
            #if i > 0:#放入lstm1和2裡面
                #tf.get_variable_scope().reuse_variables()
            ###2.lstm1 layer
            with tf.variable_scope("LSTM1"):
                if i > 0:
                   tf.get_variable_scope().reuse_variables()
                output1, state1 = self.lstm1(image_emb[:,i,:], state1)
                
            ###3. lstm2 layer
            with tf.variable_scope("LSTM2"):
                if i > 0:
                  tf.get_variable_scope().reuse_variables()
                output2, state2 = self.lstm2(tf.concat([padding, output1],axis=1), state2)
###################################二之二、Decoding Stage ######################################
        for i in range(0, self.n_caption_lstm_step): ## Phase 2 => only generate captions
            #####1.查詢現在這個字的embedding   
            #tf.get_variable_scope().reuse_variables()        
            with tf.device("/cpu:0"):#指定cpu
                #tf.get_variable_scope().reuse_variables()@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@先刪這行
                current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i]) 
            ##2.重複使用參數??放到current_embed內??
            #tf.get_variable_scope().reuse_variables()@@@先刪這行
            ###3.lstm1 layer
            with tf.variable_scope("LSTM1"):#裡面再放一個reuse
                tf.get_variable_scope().reuse_variables()
                output1, state1 = self.lstm1(padding, state1)   
            ###4.lstm2 layer
            with tf.variable_scope("LSTM2"):#裡面再放一個reuse
                tf.get_variable_scope().reuse_variables()
                output2, state2 = self.lstm2(tf.concat([current_embed, output1],axis=1), state2)   
            ###5.把每個label的長度加一:是<bos>!!!(後面1代表軸1)
            labels = tf.expand_dims(caption[:, i+1], 1)
            ###6.擴展維度為(0..bath_size)(後面1代表軸1)
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
            #######7.結合index和label            
            concated = tf.concat([indices, labels],axis=1)
            #######8.轉成onehot vector
            #onehot_labels = tf.sparse_to_dense(concated, tf.pack([self.batch_size, self.n_words]), 1.0, 0.0)##沒有pack這個屬性
            onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)
            ########9.字做input(x)乘上weight(w)再加上bias(b)
            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
            ########10.計算每一次的cross_entropy            
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits( labels=onehot_labels,logits=logit_words)
            #########11.用mask遮掉padding的部分            
            cross_entropy = cross_entropy * caption_mask[:,i]
            #########12.加到放機率分布的list中
            probs.append(logit_words)
            ##########13.計算一個批次的cross_entropy
            current_loss = tf.reduce_sum(cross_entropy)/self.batch_size
            ###########14.算總loss
            loss = loss + current_loss
        return loss, video, video_mask, caption, caption_mask, probs
##############################二、建立模型function(end)######################################
###############################三、生成句子的function(start)沒用到!!!!!!!!!!!!!!!!!!!!#################################
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
            with tf.variable_scope("LSTM1"):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                output1, state1 = self.lstm1(image_emb[:, i, :], state1)
            with tf.variable_scope("LSTM2"):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                output2, state2 = self.lstm2(tf.concat(1, [padding, output1]), state2)
########################三之二、Decoding Stage((同上同上同上))######################################
        for i in range(0, self.n_caption_lstm_step):
            #tf.get_variable_scope().reuse_variables()#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if i == 0:
                with tf.device('/cpu:0'):
                    #tf.get_variable_scope().reuse_variables()
                    current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([1], dtype=tf.int64))
            with tf.variable_scope("LSTM1"):
                #tf.get_variable_scope().reuse_variables()
                output1, state1 = self.lstm1(padding, state1)
            with tf.variable_scope("LSTM2"):
                #tf.get_variable_scope().reuse_variables()
                output2, state2 = self.lstm2(tf.confcat(1, [current_embed, output1]), state2)
            logit_words = tf.nn.xw_plus_b( output2, self.embed_word_W, self.embed_word_b)
            #從onehot vector中取出最高機率的index
            max_prob_index = tf.argmax(logit_words, 1)[0]
            #放入生成句子的list中
            generated_words.append(max_prob_index)
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
#################################包含各種function的class(end)#############################
#=====================================================================================
# Global Parameters 檔案路徑
#=====================================================================================
#影片檔(.avi)
video_path = './MLDS_hw2_data/training_data/video/'
#feature檔(.avi.npy)
video_train_feat_path = './MLDS_hw2_data/training_data/feat/'
video_test_feat_path = './MLDS_hw2_data/testing_data/feat/'
#label檔(.json)
video_train_data_path = './MLDS_hw2_data/training_label.json'
video_test_data_path = './MLDS_hw2_data/testing_label.json'
#model檔
#model_path = './mys2vt18'
model_path = './models/'

#=======================================================================================
# Train Parameters 模型參數
#=======================================================================================
"""
#原作者的參數
dim_image = 4096
dim_hidden= 1000
n_video_lstm_step = 80
n_caption_lstm_step = 20
n_frame_step = 80
n_epochs = 1000
batch_size = 50
learning_rate = 0.0001
"""
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
#########################四、句子預處理for字彙集function(start)############################
def preProBuildWordVocab(list_sentence_iterator, word_count_threshold):#原本是設為5
    # borrowed this function from NeuralTalk
    print ('preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold))
    #1.用dict存每個字的出現次數
    word_counts = {}
    #2.記住輸入的句子數
    nsents = 0
    #3.計算句子數，每個字的出現次數
    for sentence_iterator in list_sentence_iterator:#注意要多考慮一層list!!
        for sent in sentence_iterator:
            #print('sent888888888',sent)
            nsents += 1
            #4.全轉為小寫，並用單純空白來切詞
            for w in sent.lower().split(' '):
               word_counts[w] = word_counts.get(w, 0) + 1
    #4.存放最後字彙集的list
    vocab = []
    #5.只取出大於門檻值的字
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

    print ('filtered words from %d to %d' % (len(word_counts), len(vocab)))
    #print("vocab",vocab)>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>做check
    #6.index to word列出要加上去的字
    ixtoword = {}#用dict做轉換
    ixtoword[0] = '<pad>'#補零
    ixtoword[1] = '<bos>'#句子開頭
    ixtoword[2] = '<eos>'#句子結尾
    ixtoword[3] = '<unk>'#未知沒看過的字
    #7.word to index
    wordtoix = {}#逆轉換的dict
    wordtoix['<pad>'] = 0
    wordtoix['<bos>'] = 1
    wordtoix['<eos>'] = 2
    wordtoix['<unk>'] = 3
    #8.把從句子中學到的字加進dict中
    for idx, w in enumerate(vocab):
        wordtoix[w] = idx+4
        ixtoword[idx+4] = w
    #9.把自訂義的字其出現次數設為nsents
    word_counts['<bos>'] = nsents#每個句子都會出現一次，所以和句子數一樣
    word_counts['<eos>'] = nsents#每個句子都會出現一次，所以和句子數一樣
    word_counts['<pad>'] = nsents#不一定會出現???????
    word_counts['<unk>'] = nsents#不一定會出現???????
    #10.更改bias_init_vector，依每字出現次數建一個同樣長度的array[1.0,1.0,..]??
    bias_init_vector = np.array([1.0 * word_counts[ixtoword[i]] for i in ixtoword])
    #11.算平均，作長度正規化
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    #12.再取一一對數
    bias_init_vector = np.log(bias_init_vector)
    #13.再全部減掉其中最大的值(最大值變零，其他變成負的)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    #print("bias_init_vector",bias_init_vector)>>>>>>>>>>>>>>>>>>>做check
    return wordtoix, ixtoword, bias_init_vector
#########################四、句子預處理for字彙集function(end)############################
#========================================================================================
#                                    讀檔
#========================================================================================
###############################一、讀json檔#############################################
#get_video_train_data(video_data_path, video_feat_path):
print ("讀檔read start")#用原作者的才行
#1.直接用read_json讀就可以
video_data = pd.read_json('./MLDS_hw2_data/training_label.json')
#2.新增欄位'video_path'
video_data['video_path'] = video_data.apply(lambda row: row['id'] +'.npy', axis=1)
#3.加上video_train_feat_path路徑((feature!!!!))，結合label和feature
video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(video_train_feat_path, x))
#4.檢查檔案是否存在(感覺不需要)
video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists( x ))]
#5.檢查是不是str:是list!!!
# video_data = video_data[video_data['Description'].map(lambda x: isinstance(x, str))]
#6.排序所有avi檔案名#????????????????感覺不需要?????????????????
unique_filenames = sorted(video_data['video_path'].unique())
#7.移除重覆、重新排序
train_data = video_data[video_data['video_path'].map(lambda x: x in unique_filenames)]
#return train_data
"""
#1.用read讀全部內容((label!!!!))
data_str = open('./MLDS_hw2_data/training_label.json').read()
#print("data_str",data_str)#check ok
#2.轉成pd的dataframe格式json檔
video_data = pd.read_json(data_str)
#print("video_data",video_data)#check ok
#4.新增欄位'video_path'
video_data['video_path'] = video_data.apply(lambda row: row['id']+'.npy', axis=1)
#5.加上video_train_feat_path路徑((feature!!!!))，結合label和feature
video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(video_train_feat_path, x))
#video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists(x)]#檢查檔案是否存在
#print('path',video_data['video_path'])#checkn ok#????????????????感覺不需要?????????????????

#video_data = video_data[video_data['caption'].map(lambda x: isinstance(x, str))]#檢查是不是str:是list!!!
#video_data['caption'] = video_data[video_data['caption'].map(lambda x: isinstance(x, str))]#檢查是不是str:是list!!!
#print("captions",video_data['caption'])

#6.排序所有avi檔案名#????????????????感覺不需要?????????????????
#unique_filenames = sorted(video_data['video_path'].unique())
#7.移除重覆
#train_data = video_data[video_data['video_path'].map(lambda x: x in unique_filenames)]
#print("train_data",train_data)#check ok
train_data = video_data#給train直接用!!
"""
print ("讀檔read ok!!!")
#############################二、處理caption句子部分####################################
#1.取出caption欄位中所有的值，並轉為list(一個value代表一個影片的所有答案句子)
allvideo_captions_list = list(train_data['caption'].values)
#print("captions_list",len(captions_list))#check ok
#2.轉成矩陣(是list of list!!):每個影片的答案又各自是一個list
captions_array = np.asarray(allvideo_captions_list, dtype=np.object)
#print("shape", captions.shape)#captions.shape是tuple結構:shape(1450,)
#print("caption77777", captions)#check ok
#3.一一取代掉標點符號或其他非字母字元:captions現在每個都是一個list
replaced_captions_list = []#存放經過取代後的list of list!!
for sentence_list in captions_array:
    sentence_list = map(lambda x: x.replace('.', ''), sentence_list)
    sentence_list = map(lambda x: x.replace(',', ''), sentence_list)
    sentence_list = map(lambda x: x.replace('"', ''), sentence_list)
    sentence_list = map(lambda x: x.replace('\n', ''), sentence_list)
    sentence_list = map(lambda x: x.replace('?', ''), sentence_list)
    sentence_list = map(lambda x: x.replace('!', ''), sentence_list)
    sentence_list = map(lambda x: x.replace('\\', ''), sentence_list)
    sentence_list = map(lambda x: x.replace('/', ''), sentence_list)
    sentence_list = map(lambda x: x.replace('&', ''), sentence_list)
    sentence_list = map(lambda x: x.replace(')', ''), sentence_list)
    sentence_list = map(lambda x: x.replace('(', ''), sentence_list)
    sentence_list = map(lambda x: x.replace(']', ''), sentence_list)
    sentence_list = map(lambda x: x.replace('[', ''), sentence_list)
    #print("sentence_list ?????",sentence_list)#正確，為句子
    replaced_captions_list.append(sentence_list)#把處理過後的句子放入
#4.丟到句子預處理function中切詞
print("預處理preProBuildWordVocab start")
wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(replaced_captions_list, word_count_threshold=2)#宏光設2
print("preProBuildWordVocab ok!!!")
#5.存起檔案
np.save("./wordtoix", wordtoix)
np.save('./ixtoword', ixtoword)
np.save("./bias_init_vector", bias_init_vector)
############################三、用function初始化模型################################
print("初始化 Video_Caption_Generator start")
model = Video_Caption_Generator(
        dim_image=dim_image,
        n_words=len(wordtoix),
        dim_hidden=dim_hidden,
        batch_size=batch_size,
        n_lstm_steps=n_frame_step,
        n_video_lstm_step=n_video_lstm_step,
        n_caption_lstm_step=n_caption_lstm_step,
        bias_init_vector=bias_init_vector)
print("初始化 Video_Caption_Generator ok!!!")
############################四、用function建構模型################################
#1.建構模型
print("建構模型 build_model start")
tf_loss, tf_video, tf_video_mask, tf_caption, tf_caption_mask, tf_probs = model.build_model()
print("建構模型 build_model ok!!!")
#2.使用互動式Session(會自己預設???)
sess = tf.InteractiveSession() 
#3.用saver儲存模型
# my tensorflow version is 0.12.1, I write the saver with version 1.0
#saver = tf.train.Saver(max_to_keep=100, write_version=1)#max_to_keep存的checkpoint數
saver = tf.train.Saver()#不用參數
#4.設定loss計算
train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
#train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_loss)
#5.初始化
tf.global_variables_initializer().run()
#6.一再儲存最新版本的模型????
#new_saver = tf.train.Saver()#不用參數
#new_saver.restore(sess,'./models/model-100')
#saver.restore(sess, './models/model_100')?????找不到?????????
#7.把loss結果寫入txt檔
loss_fd = open('loss.txt', 'w')
#8.用list存每次的loss
loss_to_draw = []
#========================================================================================
#                                    開始訓練!!
#========================================================================================
############################一、每個批次隨機選不同句子當正確答案################################
#current_train_data = train_data
for epoch in range(0,n_epochs):#, n_epochs
    loss_to_draw_epoch = []
    #batch_captions_list = []
    #curreplace_captions_list = []
    #1.取出index:(0,1449)
    index = list(train_data.index)
    #print("index",index)
    #2.用shuffle打亂index的順序
    np.random.shuffle(index)
    #print("index",index)#順序不同
    #3.依打亂後的index選出id/video path/caption
    train_data = train_data.ix[index]
    #print("train_data.ix[0]",train_data.ix[0])
    #4.隨機選出其中一句做本次訓練的正確句子
    current_train_data = train_data.groupby('video_path').apply(lambda x: x.iloc(np.random.choice(len(x)))[0])#沒有irow
    #5.shuffle後重新index，從0開始
    current_train_data = current_train_data.reset_index(drop=True)
    #current_train_data['caption'] = current_train_data.apply(lambda row: row['caption'][np.random.choice(len(row['caption']))],axis=1)
    #current_train_data = current_train_data.apply(lambda row: row['caption'][np.random.choice(len(row['caption']))],axis=1)
    #print ('current_train_data.ix[1449]',current_train_data.ix[1449]['caption'])#只有一句，且每次不同
    #print("reset_index",current_train_data.ix[0])#check ok
    ############################二、用zip來分批每批的訓練資料################################
    for start, end in zip(
            range(0, len(current_train_data), batch_size),
            range(batch_size, len(current_train_data), batch_size)):
        #!.記住開始時間(感覺不必要??????????????)
        start_time = time.time()
        #2.擷取出每批的row
        current_batch = current_train_data[start:end]
        #print("current_batch",current_batch)#有問題
        #3.取出其中的video_path值
        current_videos = current_batch['video_path'].values
        #4.用path去讀相應的feature的npy檔:這裡才正式讀feature!!!!!!
        current_feats = list(np.zeros((batch_size, n_video_lstm_step, dim_image)))#多加list轉換!!
        current_feats_vals = list(map(lambda vid: np.load(vid), current_videos))#多加list轉換!!
        #5.加上mask遮罩
        current_video_masks = list(np.zeros((batch_size, n_video_lstm_step)))#多加list轉換!!
        #6.記住現在讀到的video的feature和其mask
        for ind,feat in enumerate(current_feats_vals):#用ind去取
            current_feats[ind][:len(current_feats_vals[ind])] = feat#存起真正的feature
            current_video_masks[ind][:len(current_feats_vals[ind])] = 1#有真正feature的mask才為1
        #7.取出當下批次的caption欄位的值，並轉為list(一個value代表一個影片的隨機一句正確句子):一層list而已
        #batch_captions_list = np.asarray(list(current_batch['caption'].values),dtype=object)
        #batch_captions_list = []
        batch_captions_list = current_batch['caption'].values
       #print("batch_captions_listt0000",len(batch_captions_list))#長度50正確
        #print("batch_captions_listt9999",batch_captions_list)#有50句正確
        #8.一一取代掉標點符號或其他非字母字元
        curreplace_captions_list = []#用list存放經過取代後的字
        """
        for current_captions in batch_captions_list:
            #current_captions = map(lambda x: '<bos> ' + x, current_captions)#在每一字前都加bos
            current_captions = '<bos> '+current_captions#加在每一句子前
            #print("problem   bbbbb!!!!!!!!!!",current_captions)#正確!
            current_captions = current_captions.replace(',', '')
            current_captions = current_captions.replace('"', '')
            current_captions = current_captions.replace('\n', '')
            current_captions = current_captions.replace('?', '')
            current_captions = current_captions.replace('!', '')
            current_captions = current_captions.replace('\\', '')
            current_captions = current_captions.replace('/', '')
            current_captions = current_captions.replace('&', '')
            current_captions = current_captions.replace('(', '')
            current_captions = current_captions.replace(')', '')
            current_captions = current_captions.replace('[', '')
            current_captions = current_captions.replace(']', '')
            #print("problem   ttttt!!!!!!!!!!",current_captions)#正確
        """#for x in current_captions: 
        for current_captions in batch_captions_list:  
            current_captions = map(lambda x: '<bos> ' + x, current_captions)
            current_captions = map(lambda x: x.replace('.', ''), current_captions)
            current_captions = map(lambda x: x.replace(',', ''), current_captions)
            current_captions = map(lambda x: x.replace('"', ''), current_captions)
            current_captions = map(lambda x: x.replace('\n', ''), current_captions)
            current_captions = map(lambda x: x.replace('?', ''), current_captions)
            current_captions = map(lambda x: x.replace('!', ''), current_captions)
            current_captions = map(lambda x: x.replace('\\', ''), current_captions)
            current_captions = map(lambda x: x.replace('/', ''), current_captions)
            current_captions = map(lambda x: x.replace('&', ''), current_captions)
            current_captions = map(lambda x: x.replace('(', ''), current_captions)
            current_captions = map(lambda x: x.replace(')', ''), current_captions)
            current_captions = map(lambda x: x.replace('[', ''), current_captions)
            current_captions = map(lambda x: x.replace(']', ''), current_captions)
            #9.轉為list方便之後處理
            current_captions = list(current_captions)
            #print("current_captions",current_captions)#被切成一個個字母
            curreplace_captions_list.append(current_captions)
            #print("curreplace_captions_list22222",curreplace_captions_list)#被切成一個個字母
        #9.把句子全轉為小寫，並用空白切詞
        for idx, each_cap in enumerate(curreplace_captions_list):
            word = each_cap.lower().split(' ')
            #10.看預測的次數是否超過所設上限
            if len(word) < n_caption_lstm_step:#未超過就加上<eos>做結尾
                curreplace_captions_list[idx] = curreplace_captions_list[idx] + ' <eos>'
            else:#超過的話
                new_word = ''#就把多的字都切掉
                for i in range(n_caption_lstm_step-1):#多預留一個位置給<eos>
                    new_word = new_word + word[i] + ' '
                curreplace_captions_list[idx] = new_word + '<eos>'#仍然加上<eos>做結尾
        #11.記住現在批次的句子
        current_caption_ind = []
        #12.記住現在看到的句子
        for cap in curreplace_captions_list:
            #print("cap7777777777",cap)#被切成一個個字母了，不是一個word了
            #13.記住現在看到的字
            current_word_ind = []
            for word in cap.lower().split(' '):#全轉為小寫，並用空白切詞
                #14.查找wordtoix做轉換
                if word in wordtoix:
                    current_word_ind.append(wordtoix[word])
                else:#找不到就輸出<unk>
                    current_word_ind.append(wordtoix['<unk>'])
            current_caption_ind.append(current_word_ind)
        #15.做padding至預設的句子最長字數
        current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=n_caption_lstm_step)
        #print("current_caption_matrix",current_caption_matrix)
        #16.存起真正的值
        current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix), 1] ) ] ).astype(int)
        #17.記住padding的位置
        current_caption_masks = np.zeros( (current_caption_matrix.shape[0], current_caption_matrix.shape[1]) )
        #18.把真正值的部分轉為矩陣
        nonzeros = np.array( map(lambda x: (x != 0).sum() + 1, current_caption_matrix ) )##加上list!!!!!!!!!!!!!!!
        #19.只處理mask值為1的
        for ind, row in enumerate(current_caption_masks):
            row[:nonzeros[ind]] = 1
        #好像沒用到
        """
        probs_val = sess.run(tf_probs, feed_dict={
            tf_video:current_feats,
            tf_caption: current_caption_matrix
            })
        """
        #20.計算loss
        _, loss_val, probs = sess.run([train_op, tf_loss, tf_probs],
                feed_dict={
                    tf_video: current_feats,
                    tf_video_mask : current_video_masks,
                    tf_caption: current_caption_matrix,
                    tf_caption_mask: current_caption_masks
                    })
        #21.把每個epoch的loss放入list中
        loss_to_draw_epoch.append(loss_val)
        #22.印出正確答案比較
        for x in range(10, 20):
            correct_sent = ''
            for i in range(0, n_caption_lstm_step):
                word = [ixtoword[word] for word in list(current_caption_matrix[:, i])]
                correct_sent = correct_sent+word[x]+' '
            print("answer",correct_sent,'\n')
        #23.印出自己預測的句子    
            my_sent = ''
            for num in probs:
                 my_sent = my_sent+ixtoword[np.argmax(num[x])] +' '
            print(my_sent)
        #24.印出每個epoch結果
        print ('idx: ', start, " Epoch: ", epoch, " loss: ", loss_val, ' Elapsed time: ', str((time.time() - start_time)))
        loss_fd.write('epoch ' + str(epoch) + ' loss ' + str(loss_val) + '\n')
    # draw loss curve every epoch:畫圖(不需要)
    loss_to_draw.append(np.mean(loss_to_draw_epoch))
    """
    plt_save_dir = "./loss_imgs"
    plt_save_img_name = str(epoch) + '.png'
    plt.plot(range(len(loss_to_draw)), loss_to_draw, color='g')
    plt.grid(True)
    plt.savefig(os.path.join(plt_save_dir, plt_save_img_name))
    """
    #25.每十次存一個目前的model
    if  np.mod(epoch, 10) == 0:#epoch!=0 and
        print ("Epoch ", epoch, " is done. Saving the model ...")
        saver.save(sess, os.path.join(model_path, './model'), global_step=epoch)

loss_fd.close()


print("ok")
