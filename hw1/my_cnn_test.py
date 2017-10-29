import os
import re
import sys
import csv
import numpy as np
from numpy import asarray
import string#string.rpartition(str)
from collections import OrderedDict

from keras.models import load_model
import h5py
#from tqdm import tqdm
#import mmap
#import pickle
import time
from sklearn.preprocessing import StandardScaler
from keras import backend
import _pickle as cPickle
import itertools#把相同的做分組!!!做移除相鄰的重複!!!!!!!!!
#from keras.utils import np_utils
#import keras

print ("testing data...")

################################輸入argv############################### 
import sys
fbank_test_PATH = sys.argv[1]
result_csv_PATH = sys.argv[2]


########################################1,參數、變數##################
#路徑
#inputpath = sys.argv[1]
#outputpath = sys.argv[2]
#做標準化
scaler = StandardScaler()

INPUT_SIZE = 69#fbank維度
PADDING_SIZE = 31


#####################################2.自建辭典做數字到字母的轉換################
tochar = {}
tochar[0] = 'a'
tochar[1] = 'b'
tochar[2] = 'c'
tochar[3] = 'a'
tochar[4] = 'e'
tochar[5] = 'c'
tochar[6] = 'g'
tochar[7] = 'h'
tochar[8] = 'i'
tochar[9] = 'L'
tochar[10] = 'k'
tochar[11] = 'l'
tochar[12] = 'm'
tochar[13] = 'n'
tochar[14] = 'B'
tochar[15] = 'D'
tochar[16] = 'L'
tochar[17] = 'r'
tochar[18] = 's'
tochar[19] = 't'
tochar[20] = 'u'
tochar[21] = 'v'
tochar[22] = 'w'
tochar[23] = 'w'
tochar[24] = 'y'
tochar[25] = 'z'
tochar[26] = 'A'
tochar[27] = 'B'
tochar[28] = 'C'
tochar[29] = 'D'
tochar[30] = 'E'
tochar[31] = 'F'
tochar[32] = 'G'
tochar[33] = 'H'
tochar[34] = 'I'
tochar[35] = 'J'
tochar[36] = 'K'
tochar[37] = 'L'
tochar[38] = 'M'
tochar[39] = 'N'
tochar[40] = 'O'
tochar[41] = 'P'
tochar[42] = 'Q'
tochar[43] = 'L'
tochar[44] = 'S'
tochar[45] = 'T'
tochar[46] = 'U'
tochar[47] = 'K'

##################################3.讀test檔####################
data = OrderedDict()#用dict存維度資料
f = open(fbank_test_PATH + '/fbank/test.ark',"r+")
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

print ("read data ok")

###########################4.做padding###############
test_x = []
#lendict = {}#存起每個句子原始長度，方便之後處理

for seqID in data.keys():
	#lendict[seqID] = len(data[seqID])#存長度
	for index in range(len(data[seqID])):#每個frame
		padding = []#抓前後來補
		
		begin = -int((PADDING_SIZE-1)/2)#開頭
		end = int((PADDING_SIZE+1)/2)#結尾
		
		for i in range(begin, end):
			if (index + i < 0 or index + i > len(data[seqID])-1):
				pad = [0.0] * 69#不夠就補零
				padding.append(pad)
			else:#夠就直接放入
				padding.append(data[seqID][index+i])

		test_x.append(padding)

input_shape = np.array(test_x).shape
print(input_shape)

print ("padding ok")


#########################5.下載model###################
model = load_model('./your_cnn_model.h5')##########上傳github

print ("load model ok")
############################6.做預測#####################
prearr = model.predict(test_x, verbose=0)

print ("prearr ok")
########################7.預測的後處理######################
charlist = []#存起所有預測的字母，方便之後切
#轉成單一數字，再轉成字母
for eachvec in prearr:
	vectochar = tochar[np.argmax(eachvec)]
	charlist.append(vectochar)
	
#切出每個句子的預測phone sequence
resultdict = OrderedDict()
begin = 0#記住每個句子的開頭
end = 0#記住每個句子的結束

for seqID in data.keys():
	begin = end#從上一句結尾開始
	end = begin + len(data[seqID])#算這一句的結尾
	#移除重複
	nullstr = ''#轉成字串表示
	phoneseq = nullstr.join(charlist[begin:end])
	phoneseq = nullstr.join(ch for ch, _ in itertools.groupby(phoneseq))#移除同組的重複
	#去掉頭尾的sil
	if ( len(phoneseq) != 0 ):#不為空
		if (phoneseq[0] == 'L' ):#切除開頭的sil
			phoneseq = phoneseq[1:]
		
		if (phoneseq[-1] == 'L'):#切除尾部的sil
			phoneseq = phoneseq[:-1]
	
	resultdict[seqID] = phoneseq#放入dict

print ("resultdict ok")
#######################8.開啟csv檔寫入#####################
#f = open("kaggle2.csv", 'w', newline='')
f = open(result_csv_PATH, 'w', newline='')
csvCursor = csv.writer(f)
#寫入標頭
#f.write('id,phone_sequence\n')
csvHeader = ['id', 'phone_sequence']#寫入標頭
csvCursor.writerow(csvHeader)
#一行行寫入內容
for seqID in resultdict.keys():
    csvdata = [seqID, resultdict[seqID]]
    #f.write(seqID+','+resultdict[seqID]+'\n')
    csvCursor.writerow(csvdata)
f.close()

print ("test ok")
