#-*- coding:utf-8 -*-
import re
import numpy as np
import pandas as pd
import os
# import sys
# reload(sys)
# sys.setdefaultencoding("gbk")
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
# 指定第一块GPU可用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)

KTF.set_session(sess)

#数据预处理
# 最大句子长度
maxlen = 32
s = open('msr_train.txt').read().decode('gbk')
s = s.split('\n')


# print(s[3])
def clean(s):  # 整理一下数据，有些不规范的地方
    if u'“/s' not in s:
        return s.replace(u' ”/s', '')
    elif u'”/s' not in s:
        return s.replace(u'“/s ', '')
    elif u'‘/s' not in s:
        return s.replace(u' ’/s', '')
    elif u'’/s' not in s:
        return s.replace(u'‘/s ', '')
    else:
        return s


s = u''.join(map(clean, s))
s = re.split(u'[，。！？、]/[bems]', s)  # 按标点符号进行切分

data = []  # 生成训练样本
label = []


def get_xy(s):  # 将字与标注分离
    s = re.findall('(.)/(.)', s)
    if s:
        s = np.array(s)
        return list(s[:, 0]), list(s[:, 1])


a, b = get_xy(s[0])
# print a[0].encode('utf8').decode('utf8')
for i in s:
    x = get_xy(i)
    if x:
        data.append(x[0])
        label.append(x[1])

d = pd.DataFrame(index=range(len(data)))
d['data'] = data
d['label'] = label
d = d[d['data'].apply(len) <= maxlen]
d.index = range(len(d))
#print d['data']
tag = pd.Series({'s': 0, 'b': 1, 'm': 2, 'e': 3, 'x': 4})
chars = []  # 统计所有字，跟每个字编号
for i in data:
    chars.extend(i)

chars = pd.Series(chars).value_counts()

chars[:] = range(1, len(chars) + 1)


# print chars

if(os.path.exists('./data/cnn_bilstm_crf_model.h5')==False):
    print 'train'
    embedding_dim = 64
    sequence_length = 32
    dropout=0.2
    #生成适合模型输入的格式
    from keras.utils import np_utils
    from sklearn.cross_validation import train_test_split
    #填充
    d['x'] = d['data'].apply(lambda x: np.array(list(chars[x])+[0]*(sequence_length-len(x))))
    d['y'] = d['label'].apply(lambda x: np.array(map(lambda y:np_utils.to_categorical(y,5), tag[x].reshape((-1,1)))+[np.array([[0,0,0,0,1]])]*(sequence_length-len(x))))
    # x=np.array(list(d['x']))
    # y=np.array(list(d['y'])).reshape(shape=(-1,sequence_length,5))
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    # 设计模型
    from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional,Reshape,Convolution2D,MaxPooling2D,merge,Dropout
    from keras.layers import ChainCRF
    from keras.callbacks import ModelCheckpoint
    from keras.models import Model
    #cnn_提取特征，使用3元
    cnn_inputs = Input(shape=(maxlen,), dtype='int32',name='cnn_inputs')
    cnn_embedding = Embedding(output_dim=embedding_dim, input_dim=len(chars)+1, input_length=sequence_length)(cnn_inputs)
    cnn_reshape = Reshape((sequence_length, embedding_dim, 1))(cnn_embedding)
    cnn_conv = Convolution2D(nb_filter=100, nb_row=3, nb_col=embedding_dim, border_mode='valid',activation='relu')(cnn_reshape)
    cnn_max_pooling=MaxPooling2D(pool_size=(int(cnn_conv.get_shape()[1]), 1))(cnn_conv)

    #BiLSTM-CRF
    bilstm_inputs=Input(shape=(maxlen,),dtype='int32',name='bilstm_inputs')
    bilstm_embedding=Embedding(len(chars)+1,embedding_dim,input_length=sequence_length,mask_zero=True)(bilstm_inputs)
    #合并cnn提取特征和普通字向量
    total_emb = merge([bilstm_embedding, cnn_max_pooling], mode='concat', concat_axis=2, name='total_emb')
    emb_droput = Dropout(dropout)(total_emb)
    #blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(emb_droput)
    blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(bilstm_embedding)
    drop=Dropout(dropout)(blstm)
    output = TimeDistributed(Dense(5))(drop)
    crf = ChainCRF()
    crf_output = crf(output)
    model = Model(input=[bilstm_inputs,cnn_inputs], output=crf_output)
    # checkpoint = ModelCheckpoint('./model/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1,
    #                             save_best_only=True, mode='auto')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    batch_size = 1024
    history = model.fit([np.array(list(d['x'])),np.array(list(d['x']))], np.array(list(d['y'])).reshape((-1,maxlen,5)), batch_size=batch_size,
                        nb_epoch=50)
    model.save('./model/cnn_bilstm_crf_model.h5')
