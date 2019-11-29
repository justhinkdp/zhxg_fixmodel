# -*- coding:utf-8 -*-
import lightgbm as lgb
import jieba
import re
import numpy as np
import string
# import os
# import cPickle as pickle


path = './data/combin_6/'
keyword_path = './keyword/'

# sentence是词典，存储要预测是语句
def key_cv(sentence):
    word2id = {}
    word2id_cat = {}
    word2id_cat_m = {}

    # mergerate=0.05

    # tfidf keyword
    for line in open(path + "keywords_single_250.txt",encoding='UTF-8'):
        for w in line.split():
            word2id[w.strip()] = len(word2id)
    ct = 0
    counts = [0, 0, 0, 0, 0]

    for line in open(path + "keywords_single_250.txt",encoding='UTF-8'):
        for w in line.split():
            word2id_cat[w.strip()] = ct
            counts[ct] += 1
        ct += 1

    for i in range(16, 21):
        for line in open(keyword_path + str(i) + ".csv",encoding='UTF-8'):
            for w in line.split():
                word2id[w.strip()] = len(word2id)

    for i in range(16, 21):
        for line in open(keyword_path + str(i) + ".csv",encoding='UTF-8'):
            for w in line.split():
                word2id_cat[w.strip()] = i - 16
                # word2id_cat_m[w.strip().decode("utf-8")]=ct
                word2id_cat_m[w.strip()] = i - 16

    data = []
    # t1 = open('D:\CodeProject\PythonProject\\nlp_zhxg\\'+testfile+'.txt')  # 打开头目录的要预测的文件，因为是第一次打开，还没有预测后剩下的other文件
    # d1 = t1.read().split('\r')

    # print len(d1)
    # t1.close()
    d1 = sentence
    kdr = []
    for s in d1:
        if d1[s] == '':
            continue
        # content = d1[s].split('|')[2] # s是key，content是需要预测的语句
        content = d1[s]
        m = re.findall('[\d]+\.wav[\d|！|_|。]+', content)
        for mm in m:
            content = content.replace(mm, '')
        content = re.sub('[\d]+_[\d]+_', '', content)
        tp = [0] * (len(word2id) + 5)
        kd = [set(), set(), set(), set(), set(), set()]

        # for key in word2id_cat_m:
        #     # 如果key直接在输入语句中
        #     if key in d1[s]:
        #         kd[word2id_cat_m[key]].add(key)
        #         continue
        #     # 如果key是需要做匹配的关键词
        #     if '?(' in key:
        #         before = key.split('?')[0]
        #         after = key.split(')')[1]
        #         if before in d1[s] and after in d1[s]:
        #             length = (d1[s].find(after) - d1[s].find(before) - before.__len__()) / 3
        #
        #             start = key.find('(')
        #             end = key.find(')')
        #             child_s = key[start + 1:end]
        #             num = string.atoi(child_s)
        #             if length < num:
        #                 kd[word2id_cat_m[key]].add(key)
        #
        # kdr.append(kd)

        for w in jieba.cut(content):
            for key in word2id:
                if w in key:
                    tp[word2id[key]] += 1
            for key in word2id_cat:
                if w in key:
                    tp[-(word2id_cat[key] + 1)] += 1
            for key in word2id_cat_m:
                if w in key and w != '?':
                    kd[word2id_cat_m[key]].add(key)

        kdr.append(kd)

        data.append(tp)

    # 处理后得到数组data进行预测
    data = np.array(data)
    # print data.shape
    r = []
    for i in range(5):
        clf = lgb.Booster(model_file=path + "models/key_cv" + str(i) + ".m")
        if len(r) == 0:
            r = clf.predict(data)
        else:
            r += clf.predict(data)

    # rr=['融合离网]', '[离网]', '[融合拆分]', '[智能组网]', '[宽带无资源]','[其他]']
    rr = ['[CFLH-RHLW]', '[CFLH-LWALL]', '[CFLH-RHCF]', '[JZ-ZNZW]', '[WTWJ-KDWZY]','[others]']
    ct = 0
    ct2 = 0
    oth = 0
    cor = 0
    prers = []
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    tow = open('result.txt', 'a',encoding='UTF-8')
    # townext = open(path + testfile+'_others.txt', 'wb')

    # 将预测结果r与原始数据文字部分d1打包，即r与d1一一对应,d1为词典，在这里v[1]是词典的key
    for v in list(zip(r,d1,kdr)):
        # tpr=np.where(v[0][:-1]==max(v[0][:-1]))[0][0]
        # if v[0][-1]>v[0][tpr]*3:
        #     tpr=-1
        tpr=np.where(v[0][:]==max(v[0][:]))[0][0]
        b = np.argsort(np.array(list(v[0])))
        value = ""
        for index in range(0, b.__len__())[::-1]:
            # value +="|"+ rr[b[index]]+"|"+str(round(v[0][b[index]],2)) +"|"+" ".join(list(v[2][b[index]])).encode("utf-8")
            value +="$$"+ rr[b[index]]+"|"+str(round(v[0][b[index]],2)) +"|"+"|".join(list(v[2][b[index]]))
        write_str = "|" + str(v[1]) + value + "\n"
        tow.write(write_str)
        # tow.write(rr[tpr]+"|"+v[1].strip()+"|"+" ".join(list(v[2][tpr])).encode("utf-8")+"\n")
        # if rr[tpr] == "[others]":
        #     townext.write(v[1].strip()+"\r")
        # else:
        #     tow.write(rr[tpr]+"|"+v[1].strip()+"\n")
        ct+=1
        if ct<128 and tpr!=1:
            ct2+=1
        elif ct>128 and tpr==1:
            ct2+=1
        # print rr[tpr]
        prers.append(rr[tpr])
        if tpr == 2:
            cor+=1
    tow.flush()
    tow.close()


    # print '---------------6 end'