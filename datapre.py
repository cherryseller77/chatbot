# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import string
import re
import json
import torch
import torch.nn as nn
from functools import reduce
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



stopwords = set(stopwords.words('english'))
questionword = set(['what','where','when','how','who','which'])

def read_corpus(path):
    """
    读取给定的语料库，并把问题列表和答案列表分别写入到 qlist, alist 里面。 在此过程中，不用对字符换做任何的处理（这部分需要在 Part 2.3里处理）
    qlist = ["问题1"， “问题2”， “问题3” ....]
    alist = ["答案1", "答案2", "答案3" ....]
    务必要让每一个问题和答案对应起来（下标位置一致）
    """
    
    with open(path,'r',encoding="utf8") as f:
        all_data = json.loads(f.read())
    data = all_data["data"]
    qlist = []
    alist = []
    
    for dic in data:
        paragraphs = dic["paragraphs"]
        for para in paragraphs:
            qas = para["qas"]
            for qa in qas:
                if qa["answers"] != []:
                    answer = qa["answers"][0]["text"]
                    alist.append(answer)
                    question = qa["question"]
                    qlist.append(question)
    assert len(qlist) == len(alist)  # 确保长度一样
    return qlist, alist

def segmentWords(lst):
    total = 0
    word_dict = {}
    for line in lst:
        pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
        sentence = pattern.sub("", line)
        words = sentence.split()
        for word in words:
            word_dict[word] = word_dict.get(word, 0) + 1
            total += 1
    return total,word_dict


#path = "G:\\QA\\question.json"
#qlist, alist = read_corpus(path)



#word_total,q_dict = segmentWords(qlist)
#total_diff_word = len(q_dict.keys())
#print("总共 %d 个单词" % word_total)
#print("总共 %d 个不同的单词" % total_diff_word)
#stemmer = PorterStemmer()
#stopwords = set(stopwords.words('english'))
#questionword = set(['what','where','when','how','who','which'])
 
# 预处理：去标点符号，去停用词，stemming,将数字转换为'#number'表示
def preprocessing(lst,questionword,stopwords):
    new_list=[]
    word_dic = {}
    stemmer = PorterStemmer()
    for line in lst:
        pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
        sentence = pattern.sub("", line)
        sentence = sentence.lower()
        words = sentence.split()
        temp = []
        for word in words:
            if word in questionword:
                temp.append(word)
            elif word not in stopwords:
                word = "#number" if word.isdigit() else word
                w = stemmer.stem(word)
                word_dic[w] = word_dic.get(w, 0) + 1
                temp.append(w)
        new_list.append(temp)
    return word_dic,new_list
 
# 画出100分为类的词频统计图
def drawgraph(dic, name):
    freq = list(dic.values())
    freq.sort(reverse=True)
    temp = [n for n in freq if n <=50]
    plt.plot(range(len(temp)),temp,'r-')
    plt.ylabel(name)
    plt.show()
 
# 过滤掉词频低于2，大于10000的词
def filterword(dic, lst,questionword, bottom,top):
    temp = []
    for k,v in dic.items():
        if v >= bottom and v <= top:
            temp.append(k)
    new_list = []
    for line in lst:
        words = [w for w in line if w in temp or w in questionword]
        new_list.append(' '.join(words))
                
    return new_list
 
#q_dict,q_list = preprocessing(qlist)
#drawgraph(q_dict,"word frequency of qlist")
# 
#a_dict,a_list = preprocessing(alist)
#drawgraph(a_dict, "word frequency of alist")
# 
#new_qlist = filterword(q_dict, q_list, 2, 10000)
#new_alist = filterword(a_dict, a_list, 2, 10000)
##print("the length of new alist is ",len(new_alist))
##print("the length of new qlist is ",len(new_qlist))
#
#
#
#vectorizer =  TfidfVectorizer()# 定一个tf-idf的vectorizer
#
#X = vectorizer.fit_transform(new_qlist)  # 结果存放在X矩阵
 
#   计算tf-idf的稀疏度
#x_mat = X.toarray()
#n = len(x_mat)
#m = len(x_mat[0])
#t = 0
#for i in range(n):
#    for j in range(m):
#        if x_mat[i][j] != 0:
#            t += 1
#sparsity = t / (n*m)
#print (sparsity)  # 打印出稀疏度

def top5results(input_q):
    """
    给定用户输入的问题 input_q, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
    1. 对于用户的输入 input_q 首先做一系列的预处理，然后再转换成tf-idf向量（利用上面的vectorizer)
    2. 计算跟每个库里的问题之间的相似度
    3. 找出相似度最高的top5问题的答案
    """
    # 问题预处理
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    sentence = pattern.sub("", input_q)
    sentence = sentence.lower()
    words = sentence.split()
    result = []
    for word in words:
        if word not in stopwords:
            word = "#number" if word.isdigit() else word
            w = stemmer.stem(word)
            result.append(w)
    
    #计算相似度
    input_seg = ' '.join(result)
    input_vec = vectorizer.transform([input_seg])
    res = cosine_similarity(input_vec, X)[0]
    
    #得到top 5的索引
    score_idx = dict((i,v) for i,v in enumerate(res))
    score_sorted = sorted(score_idx.items(), key=lambda k:k[1],reverse=True)
    idx = [idx[0] for idx in score_sorted]
    top_idxs = idx[:5]  # top_idxs存放相似度最高的（存在qlist里的）问题的下表 
    print(top_idxs)
    
    result = [alist[i] for i in top_idxs]
    return result  # 返回相似度最高的问题对应的答案，作为TOP5答案  



#print (top5results("when did Beyonce start becoming popular"))
#print (top5results("what languge does the word of 'symbiosis' come from"))


def inverter(new_qlist):
    inverted_idx = {}  # 定一个一个简单的倒排表
    for i in range(len(new_qlist)):
        for word in new_qlist[i].split():
            if word not in inverted_idx:
                inverted_idx[word] = [i]
            else:
                inverted_idx[word].append(i)
    for k in inverted_idx:
        inverted_idx[k] = sorted(inverted_idx[k])
    return inverted_idx

    

def intersections(candidates):
    n = len(candidates)
    result =[]
    for i in range(n-1):
        for j in range(i+1,n):
            result += (list(candidates[i].intersection(candidates[j])))
            
    result = list(set(result))
    return result
    
        
        
def top20results_invidx(input_q,vectorizer,X,inverted_idx,qlist):
    """
    给定用户输入的问题 input_q, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
    1. 利用倒排表来筛选 candidate
    2. 对于用户的输入 input_q 首先做一系列的预处理，然后再转换成tf-idf向量（利用上面的vectorizer)
    3. 计算跟每个库里的问题之间的相似度
    4. 找出相似度最高的top5问题的答案
    """
    # 问题预处理
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    sentence = pattern.sub("", input_q)
    sentence = sentence.lower()
    words = sentence.split()
    result = []
    stemmer = PorterStemmer()
    for word in words:
        if word not in stopwords:
            word = "#number" if word.isdigit() else word
            w = stemmer.stem(word)
            result.append(w)
    
    # 根据倒排表筛选出候选问题索引
    candidates = []
    for word in result:
        if word in inverted_idx:
            ids = inverted_idx[word]
            candidates.append(set(ids))
    candidate_idx = intersections(candidates)  # 候选问题索引
#    return candidate_idx
    
    input_seg = ' '.join(result)
    input_vec = vectorizer.transform([input_seg])
    
    # 与每个候选问题计算相似度
    res = []
    for i in candidate_idx:
        score = cosine_similarity(input_vec,X[i])[0]
        res.append((i,score[0]))
    res_sorted = sorted(res,key=lambda k:k[1],reverse=True)
#    print(res_sorted)
    
    # 根据索引检索top 5答案
#    answers = []
    questions = []
    questions_id = []
    i = 0
    for (idx,score) in res_sorted:
        if i < 20:
            answer = qlist[idx]
            questions.append(answer)
            questions_id.append(idx)
        i += 1
    
    return questions,questions_id





