# -*- coding:utf-8 -*-
import lightgbm as lgb
import jieba
import re
import numpy as np
import string
# import os
# import cPickle as pickle
import predict_ZQJ_3_2
import importlib

# import sys
# importlib.reload(sys)
# sys.setdefaultencoding('utf-8')


path = './data/combin_1/'
keyword_path = './keyword/'
# testfile = "workordercontent_20180305_all"
# testfile = "minitest0"

# sentence是词典，存储要预测是语句
def key_cv(sentence):
    word2id={}
    word2id_cat={}
    word2id_cat_m = {}

    # mergerate=0.05

    # tfidf keyword
    for line in open(path+"keywords_single_250.txt",encoding='UTF-8'):
        for w in line.split():
            word2id[w.strip()] = len(word2id)
    ct = 0
    counts = [0,0,0]

    for line in open(path+"keywords_single_250.txt",encoding='UTF-8'):
        for w in line.split():
            word2id_cat[w.strip()]=ct
            counts[ct] += 1
        ct += 1

    for i in range(1,4):
        for line in open(keyword_path + str(i)+".csv",encoding='UTF-8'):
            for w in line.split():
                word2id[w.strip()] = len(word2id)

    for i in range(1,4):
        for line in open(keyword_path +str(i)+".csv",encoding='UTF-8'):
            for w in line.split():
                word2id_cat[w.strip()]=i-1
                # word2id_cat_m[w.strip().decode("utf-8")]=ct
                word2id_cat_m[w.strip()]=i-1

    data=[]
    # t1 = open('D:\CodeProject\PythonProject\\nlp_zhxg\\'+testfile+'.txt')  # 打开头目录的要预测的文件，因为是第一次打开，还没有预测后剩下的other文件
    # d1 = t1.read().split('\r')

    # print len(d1)
    # t1.close()
    d1 = sentence
    kdr = []

    for s in d1:
        if d1[s] == '':
            continue
        #content = d1[s].split('|')[2] # s是key，content是需要预测的语句
        content = d1[s]
        m = re.findall('[\d]+\.wav[\d|！|_|。]+', content)
        for mm in m:
            content = content.replace(mm, '')
        content = re.sub('[\d]+_[\d]+_', '', content)
        tp=[0]*(len(word2id)+3)

        # 四个set表示四个类别中有哪些关键词在这个语句中命中
        kd = [set(), set(), set(), set()]

        # # 找关键词是否在句子中，把该关键词存放在kd对应类别的set中
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

        data.append(tp)
        kdr.append(kd)

    # 处理后得到数组data进行预测
    data=np.array(data)
    # print data.shape
    r=[]
    for i in range(5):
        clf = lgb.Booster(model_file=path+"models/key_cv" + str(i) + ".m")
        if len(r)==0:
            r=clf.predict(data)
        else:
            r+=clf.predict(data)

    # rr=['[融合新装]', '[租机续约]', '[套餐迁转]', '[其他]']
    rr = ['[XZ-RHXZ]', '[XYXF-ZJXY]', '[TCQY-BXLLTCQY]', '[others]']
    ct=0
    ct2=0
    oth = 0
    cor = 0
    prers = []
    count1 = 0
    count2 = 0
    count3 = 0
    count4 =0
    tow = open('result.txt', 'w',encoding='UTF-8')
    # townext = open(path + testfile+'_others.txt', 'wb')

    # 将预测结果r与原始数据文字部分d1打包，即r与d1一一对应,d1为词典，在这里v[1]是词典的key
    for v in list(zip(r,d1,kdr)):
        tpr=np.where(v[0][:]==max(v[0][:]))[0][0]
        b = np.argsort(np.array(list(v[0])))
        if rr[tpr] == "[others]":
            # townext.write(v[1].strip()+"\r")
            count4 +=1
        else:
            value = ""
            for index in range(0, b.__len__())[::-1]:
                value +="$$"+ rr[b[index]]+"|"+str(round(v[0][b[index]],2)) +"|"+"|".join(list(v[2][b[index]]))
            # value += "$$" + rr[b[tpr]] + "|" + str(round(v[0][b[tpr]], 2)) + "|" + "|".join(list(v[2][b[tpr]])).encode("utf-8")
            #tow.write("|".join(v[1].strip().split("|")[0:2]) + value + "\n")

            write_str = "|"+str(v[1])+ value + "\n"
            tow.write(write_str)
            # 直接输出，输出为：种类+语句
            # print rr[tpr] + "|" + str(round(v[0][tpr], 2)) + "|" + v[1].strip() + d1[v[1]]
            # 删掉预测过的语句
            del d1[v[1]]

            if tpr == 0:
                count1 +=1
            elif tpr == 1:
                count2 +=1
            elif tpr ==2:
                count3 +=1
        # # tpr=np.where(v[0][:-1]==max(v[0][:-1]))[0][0]
        # # if v[0][-1]>v[0][tpr]*3:
        # #     tpr=-1
        # tpr=np.where(v[0][:]==max(v[0][:]))[0][0]
        # b = np.argsort(np.array(list(v[0])))
        # if rr[tpr] == "[others]":
        #     townext.write(v[1].strip()+"\r")
        # else:
        #     for index in range(0, b.__len__())[::-1]:
        #         tow.write(rr[b[index]]+"|"+str(round(v[0][b[index]],2))+"|"+v[1].strip()+"\n")
        ct+=1
        if ct<128 and tpr!=1:
            ct2+=1
        elif ct>128 and tpr==1:
            ct2+=1
        # print rr[tpr]
        prers.append(rr[tpr])
        if tpr == 2:
            cor+=1
    # print cor, len(d1)
    '''
    print 'print left sentence'
    for i in d1:
        print i + d1[i]
    '''
    tow.flush()
    tow.close()
    # 如果d1不为空，继续预测
    # print '---------------1 end'
    if d1:
        predict_ZQJ_3_2.key_cv(d1)
    else:
        print("预测完成")











'''
def key_cv_ig():
    word2id = {}
    word2id_cat = {}

    # mergerate=0.05

    # tfidf keyword
    for line in open(path + "keywords_single_250.txt"):
        for w in line.split():
            word2id[w.strip().decode("utf-8")] = len(word2id)
    ct = 0
    counts = [0, 0, 0]

    for line in open(path + "keywords_single_250.txt"):
        for w in line.split():
            word2id_cat[w.strip().decode("utf-8")] = ct
            counts[ct] += 1
        ct += 1

    # IG特征
    igkey = pickle.load(open(path + 'IG/igwords.pkl', 'rb'))[:100]
    for key in igkey:
        if word2id.has_key(key[0]):
            continue
        word2id[key[0]] = len(word2id)

    for i in range(1, 4):
        for line in open("./data/" + str(i) + ".csv"):
            for w in line.split():
                word2id[w.strip().decode("utf-8")] = len(word2id)

    for i in range(1, 4):
        for line in open("./data/" + str(i) + ".csv"):
            for w in line.split():
                word2id_cat[w.strip().decode("utf-8")] = i - 1

    data = []
    t1 = open('./'+testfile+'.txt')
    d1 = t1.read().split('\r')

    print len(d1)
    t1.close()
    for s in d1:
        if s == '':
            continue
        content = s.split('|')[2]
        m = re.findall('[\d]+\.wav[\d|！|_|。]+', content)
        for mm in m:
            content = content.replace(mm, '')
        content = re.sub('[\d]+_[\d]+_', '', content)
        tp = [0] * (len(word2id) + 3)
        for w in jieba.cut(content):
            for key in word2id:
                if w in key:
                    tp[word2id[key]] += 1
            for key in word2id_cat:
                if w in key:
                    tp[-(word2id_cat[key] + 1)] += 1
        data.append(tp)

    data = np.array(data)
    print data.shape
    r = []
    for i in range(5):
        clf = lgb.Booster(model_file=path + "/models/key_ig_cv" + str(i) + ".m")
        if len(r) == 0:
            r = clf.predict(data)
        else:
            r += clf.predict(data)

    # rr=['[融合新装]', '[租机续约]', '[套餐迁转]', '[其他]']
    rr = ['[XZ-RHXZ]', '[XFXY-ZJXY]', '[TCQY-BXLLTCQY]', '[others]']
    ct = 0
    ct2 = 0
    oth = 0
    cor = 0
    prers = []
    tow = open(testfile+'_3_rs3.txt', 'wb')
    townext = open(testfile+'_others_ig.txt', 'wb')
    for v in zip(r, d1):
        tpr = np.where(v[0][:-1] == max(v[0][:-1]))[0][0]
        if v[0][-1] > v[0][tpr] * 3:
            tpr = -1
        # print rr[tpr]
        tpr = np.where(v[0][:] == max(v[0][:]))[0][0]
        # tow.write(rr[tpr] + "|" + v[1].strip() + "\n")
        if rr[tpr] == "[others]":
            townext.write(v[1].strip()+"\r")
        else:
            tow.write(rr[tpr]+"|"+v[1].strip()+"\n")
        ct += 1
        if ct < 128 and tpr != 1:
            ct2 += 1
        elif ct > 128 and tpr == 1:
            ct2 += 1
        print rr[tpr]
        prers.append(rr[tpr])
        if tpr == 2:
            cor += 1
    print cor, len(d1)

'''
