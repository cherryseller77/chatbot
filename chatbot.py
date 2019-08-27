# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 21:07:26 2019

@author: peng
"""
from datapre import *

from QA import *
from rerank import *

path = "G:\\QA\\question.json"
qlist, alist = read_corpus(path)





stopwords = set(stopwords.words('english'))  #加载停用词
questionword = set(['what','where','when','how','who','which']) #加载疑问词



q_dict,q_list = preprocessing(qlist,questionword,stopwords)   #清理文本
a_dict,a_list = preprocessing(alist,questionword,stopwords)

new_qlist = filterword(q_dict, q_list,questionword, 2, 10000)  #去除频率大于10000，小于2的词
new_alist = filterword(a_dict, a_list,questionword, 2, 10000)

vectorizer =  TfidfVectorizer()# 定一个tf-idf的vectorizer
X = vectorizer.fit_transform(new_qlist)  # 结果存放在X矩阵


inverted_idx = inverter(new_qlist)
input_q = "when did Beyonce start becoming popular"

glove_path = r'G:\chabot1\glove.6B.50d.txt'
vocab,embedding = loadGlove(glove_path)
qlist, alist = read_corpus(path)

model = train(qlist)



def chatbot(input_q,model):
    question,index = top5results_invidx(input_q,vectorizer,X,inverted_idx,qlist)
    result = top5answer(input_q,question,index,alist,model)
    return result



chatbot("What does Beyonce's mother do?",model)


qlist[8154]
