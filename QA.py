# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:26:00 2019

@author: peng
"""
import numpy as np
import random
import pandas as pd
import torch.nn as nn
import torch
import re


# 获取embedding对应的index
def getindex(qlist,index,maxlen):
    res = np.zeros((len(qlist),maxlen),dtype = 'int32')
    
    raw = 0
    for q in qlist:
        i = 0
        q = re.sub("[^a-zA-Z]", " ", q) 
        words = q.split()
        for word in words:
            if i <maxlen:
                if word in index:
                    res[raw][i] = index[word]
                else:
                    res[raw][i] = random.randint(1,len(index)-1)
                i +=1
        raw +=1
    return res

# 加载glove词向量
def loadGlove(path):
    vocab = {}
    embedding = []
    vocab["UNK"] = 0
    embedding.append([0]*50)
    file = open(path, 'r', encoding='utf8')
    i = 1
    for line in file:
        row = line.strip().split()
        vocab[row[0]] = i
        embedding.append(row[1:])
        i += 1
    print("Finish load Glove")
    file.close()
    return vocab, embedding

# 将str词向量变成float
def floatembedding(embedding):
    res = []
    for line in embedding:
        res.append(list(map(lambda x: float(x),line)))
    return res

class SiameseLSTM(nn.Module):
    def __init__(self,opt,embedding):
        super(SiameseLSTM, self).__init__()
        self.layer = opt.layer
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=True)
        self.lstm = nn.LSTM(input_size = opt.embedding_size,hidden_size=opt.hidden_size, \
                            num_layers = opt.layer,bias = True,batch_first = True,bidirectional =True)
        
        self.fc = nn.Sequential(nn.Linear(2,2),
                                 nn.Softmax(dim=-1))
    def forward(self,q,a):
        q_,a_ = self.embedding(q),self.embedding(a)
        q_out,(h1,c1) = self.lstm(q_)
        a_out,(h2,c2) = self.lstm(a_)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        c = abs(cos(h1[-1],h2[-1]))
        res = torch.exp(-torch.sum(torch.abs(h1[-1]-h2[-1]),1))
        encoding = torch.stack((c,res),dim=1)
        output = self.fc(encoding)
        return output

    
class args():
    def __init__(self):
        self.layer = 2
        self.embedding_size = 50
        self.hidden_size = 32


def f1score(TP,FP,FN):
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    f1 = 2*recall*precision/(precision+recall)
    print('训练集分数')
    print(f1)
    print(TP,FP,FN)
    
def Evalue(y_pred,y_target):
    TP,FP,FN=0,0,0
    y_pred = torch.argmax(y_pred,dim =1)
    for pre,tar in zip(y_pred,y_target):
        if int(pre) == 1 and int(tar.item()) ==1:
            TP += 1
        elif int(pre) == 1 and int(tar.item()) ==0:
            FP += 1
        elif int(pre)==0 and int(tar.item()) ==1:
            FN += 1
    return TP,FP,FN     
        
def train(qlist):
    glove_path = r'G:\chabot1\glove.6B.50d.txt'
    vocab,embedding = loadGlove(glove_path)
    
    question = qlist+qlist
    answer = qlist.copy()
    random.shuffle(qlist)
    answer = answer+qlist
    label = [[1]*len(qlist)+[0]*len(qlist)][0]           
                
    question_index = getindex(question,vocab,12)       
    answer_index = getindex(answer,vocab,12)
    
    question_input = torch.LongTensor(question_index)
    answer_input = torch.LongTensor(answer_index)
    label_input = torch.LongTensor(label)
    
    batchSize = 32
    train_set = torch.utils.data.TensorDataset(question_input,answer_input,label_input)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batchSize,
                                             shuffle=True)
    
    
    opt = args()
    embedding = floatembedding(embedding)
    embedding = torch.FloatTensor(embedding)      
    
    model = SiameseLSTM(opt,embedding)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            
    epoch = 5
    print('start training')
    for i in range(epoch):
        TP,FP,FN = 0,0,0
        for q,a,label in train_iter:
    #        q,a,label = q.cuda(),a.cuda(),label.cuda()
            model.zero_grad()
            score = model(q,a)
    #        print(score)
            loss = loss_function(score, label)
            loss.backward()
            optimizer.step()
            
            a,b,c = Evalue(score,label)
            TP += a
            FP += b
            FN += c
        f1score(TP,FP,FN)
            
    return model


