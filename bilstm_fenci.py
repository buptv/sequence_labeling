#-*- coding:utf-8 -*-
import re
import numpy as np
import pandas as pd
import os
# import sys
# reload(sys)
# sys.setdefaultencoding("gbk")

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

if(os.path.exists('./my_model.h5')==False):
    #生成适合模型输入的格式
    from keras.utils import np_utils
    d['x'] = d['data'].apply(lambda x: np.array(list(chars[x])+[0]*(maxlen-len(x))))
    d['y'] = d['label'].apply(lambda x: np.array(map(lambda y:np_utils.to_categorical(y,5), tag[x].reshape((-1,1)))+[np.array([[0,0,0,0,1]])]*(maxlen-len(x))))

    #设计模型
    word_size=64
    maxlen=32
    from keras.layers import Dense,Embedding,LSTM,TimeDistributed,Input,Bidirectional
    from keras.models import Model

    sequence=Input(shape=(maxlen,),dtype='int32')
    embedded=Embedding(len(chars)+1,word_size,input_length=maxlen,mask_zero=True)(sequence)
    blstm=Bidirectional(LSTM(64,return_sequences=True),merge_mode='sum')(embedded)
    output=TimeDistributed(Dense(5,activation='softmax'))(blstm)
    model=Model(input=sequence,output=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    batch_size=1024
    history = model.fit(np.array(list(d['x'])), np.array(list(d['y'])).reshape((-1,maxlen,5)), batch_size=batch_size, nb_epoch=50)
    model.save('my_model.h5')

from keras.models import load_model
model=load_model('my_model.h5')

#转移概率，单纯用了等概率
zy = {'be':0.5,
      'bm':0.5,
      'eb':0.5,
      'es':0.5,
      'me':0.5,
      'mm':0.5,
      'sb':0.5,
      'ss':0.5
     }

zy = {i:np.log(zy[i]) for i in zy.keys()}

def viterbi(nodes):
    paths = {'b':nodes[0]['b'], 's':nodes[0]['s']}
    for l in range(1,len(nodes)):
        paths_ = paths.copy()
        paths = {}
        for i in nodes[l].keys():
            nows = {}
            for j in paths_.keys():
                if j[-1]+i in zy.keys():
                    nows[j+i]= paths_[j]+nodes[l][i]+zy[j[-1]+i]
            k = np.argmax(nows.values())
            paths[nows.keys()[k]] = nows.values()[k]
    return paths.keys()[np.argmax(paths.values())]

def simple_cut(s):
    if s:
        #选择未填补部分
        r = model.predict(np.array([list(chars[list(s)].fillna(0).astype(int))+[0]*(maxlen-len(s))]), verbose=False)[0][:len(s)]
        r = np.log(r)
        nodes = [dict(zip(['s','b','m','e'], i[:4])) for i in r]
        #返回维特比最大可能路径（序列）
        t = viterbi(nodes)
        #print "vbt return:"+t
        words = []
        for i in range(len(s)):
            if t[i] in ['s', 'b']:
                words.append(s[i])
            else:
                words[-1] += s[i]
        # word_ss=""
        # fenge=" "
        # for w in words:
        #     word_ss=word_ss+w+fenge
        # print word_ss.encode('utf-8').decode('utf-8')
        return words
    else:
        return []

not_cuts = re.compile(u'([\da-zA-Z ]+)|[。，、？！\.\?,!]')
def cut_word(s):
    result = []
    j = 0
    for i in not_cuts.finditer(s):
        result.extend(simple_cut(s[j:i.start()]))
        result.append(s[i.start():i.end()])
        j = i.end()
    result.extend(simple_cut(s[j:]))
    return result

# sentence = u'人们思考问题往往不是从零开始的。就好像你现在阅读这篇文章一样，你对每个词的理解都会依赖于你前面看到的一些词，\
#       而不是把你前面看的内容全部抛弃了，忘记了，再去理解这个单词。也就是说，人们的思维总是会有延续性的。'
sentence = u'有时候我在想，傻晴为什么会那么傻，后来就没想了，可能她就是傻的没理由吧。'
result = cut_word(sentence)
fenge=" "
rss=""
for word in result:
    rss=rss+word+fenge+'/'
print rss.encode('utf-8').decode('utf-8')

