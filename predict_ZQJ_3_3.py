# -*- coding:utf-8 -*-
import lightgbm as lgb
import jieba
import re
import numpy as np
import string
# import os
# import cPickle as pickle
import predict_ZQJ_3_4

path = './data/combin_3/'
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
    counts = [0, 0, 0]

    for line in open(path + "keywords_single_250.txt",encoding='UTF-8'):
        for w in line.split():
            word2id_cat[w.strip()] = ct
            counts[ct] += 1
        ct += 1

    for i in range(7, 10):
        for line in open(keyword_path + str(i) + ".csv",encoding='UTF-8'):
            for w in line.split():
                word2id[w.strip()] = len(word2id)

    for i in range(7,10):
        for line in open(keyword_path +str(i)+".csv",encoding='UTF-8'):
            for w in line.split():
                word2id_cat[w.strip()]=i-7
                # word2id_cat_m[w.strip()]=ct
                word2id_cat_m[w.strip()]=i-7

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
        tp = [0] * (len(word2id) + 3)
        kd = [set(), set(), set(), set()]

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

    # rr=['单C新装]', '[宽带提速]', '[手机换新]', '[其他]']
    rr = ['[XZ-DCXZ]', '[TCQY-KDJS]', '[HJHKHGX-SJHX]', '[others]']
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
    for v in list(zip(r, d1,kdr)):
        tpr = np.where(v[0][:] == max(v[0][:]))[0][0]
        b = np.argsort(np.array(list(v[0])))
        if rr[tpr] == "[others]":
            # townext.write(v[1].strip()+"\r")
            count4 += 1
        else:
            value = ""
            for index in range(0, b.__len__())[::-1]:
                # value +="|"+ rr[b[index]]+"|"+str(round(v[0][b[index]],2)) +"|"+" ".join(list(v[2][b[index]])).encode("utf-8")
                value += "$$" + rr[b[index]] + "|" + str(round(v[0][b[index]], 2)) + "|" + "|".join(list(v[2][b[index]]))
            # tow.write(rr[tpr] + "|" + v[1].strip() + value + "\n")
            write_str = "|"+str(v[1])+ value + "\n"
            tow.write(write_str)
            # 直接输出，输出为：种类+语句
            # print rr[tpr] + "|" + str(round(v[0][tpr], 2)) + "|" + v[1].strip() + d1[v[1]]
            # 删掉预测过的语句
            del d1[v[1]]

            if tpr == 0:
                count1 += 1
            elif tpr == 1:
                count2 += 1
            elif tpr == 2:
                count3 += 1
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
        ct += 1
        if ct < 128 and tpr != 1:
            ct2 += 1
        elif ct > 128 and tpr == 1:
            ct2 += 1
        # print rr[tpr]
        prers.append(rr[tpr])
        if tpr == 2:
            cor += 1
    # print cor, len(d1)
    '''
    print 'print left sentence'
    for i in d1:
        print i + d1[i]
    '''
    tow.flush()
    tow.close()
    # print '---------------3 end'
    # 如果d1不为空，继续预测
    if d1:
        predict_ZQJ_3_4.key_cv(d1)
    else:
        print("预测完成")