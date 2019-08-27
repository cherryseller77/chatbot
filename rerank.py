# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 17:00:56 2019

@author: peng
"""
from QA import loadGlove
import numpy as np
import re
import random
import torch

glove_path = r'G:\chabot1\glove.6B.50d.txt'
vocab,embedding = loadGlove(glove_path)


def getinput(question,index,maxlen):
    res = np.zeros((1,maxlen),dtype = 'int32')
    q = re.sub('[^a-zA-Z]',' ',question)
    words = q.split()
    i = 0
    for word in words:
        if i <maxlen:
            if word in index:
                res[0][i] = index[word]
            else:
                res[0][i] = random.randint(1,len(index)-1)
            i +=1
    return res
        
    


def top5answer(question,qlist,indexs,alist,model):
    global vocab
    q = getinput(question,vocab,12)
    result = []
    answer = []
    score_index = dict()
    for index,s in zip(indexs,qlist):
        input_ = getinput(s,vocab,12)
        q,input_ = torch.LongTensor(q),torch.LongTensor(input_)
        score = model(q,input_)[0][1]
#        score.nn.softmax(score,dim=1)
        score_index[index] = float(score)

    score_sorted = sorted(score_index.items(), key=lambda k:k[1],reverse=True)
    idx = [idx[0] for idx in score_sorted]
    top_idxs = idx[:5]  # top_idxs存放相似度最高的（存在qlist里的）问题的下表 
    print(top_idxs)
        
    result = [alist[i] for i in top_idxs]
    print()
    return result
            
#
#
#print (top5answer("when did Beyonce start becoming popular"))
#    
#    
#    question = "when did Beyonce start becoming popular"