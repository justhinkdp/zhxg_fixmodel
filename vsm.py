# encoding:utf-8

import numpy as np
import jieba

# 构建文本向量
keyword_path = 'D:\CodeProject\PythonProject\\nlp_zhxg\keyword\\'

def vsmbuild(clabel):
    if clabel ==1:
        word2id = {}
        word2id_cat = {}
        path = 'D:\CodeProject\PythonProject\\nlp_zhxg\data\combin_1\\'

        # 处理自己生成的关键词
        for line in open(path + "keywords_single_250.txt"):
            for w in line.split():
                word2id[w.strip().decode("utf-8")] = len(word2id) # 相当于给每个feature.py提取的关键词编号1,2,3,4...，‘关键词1’：‘1’
        ct=0
        counts = [0, 0, 0]
        for line in open(path + "keywords_single_250.txt"):
            for w in line.split():
                word2id_cat[w.strip().decode("utf-8")] = ct
                # counts[ct]+=1
                # ct+=1

        # 处理给出的关键词
        for i in range(1, 4):
            for line in open(keyword_path + str(i) + ".csv"):
                for w in line.split():
                    word2id[w.strip().decode("utf-8")] = len(word2id)

        for i in range(1, 4):
            for line in open(keyword_path + str(i) + ".csv"):
                for w in line.split():
                    word2id_cat[w.strip().decode("utf-8")] = i - 1

        data = []
        paths = [path + "rhxz.txt", path + "zjxy.txt", path + "tcqy.txt", path + "others.txt"]

        for i in range(4):
            # print i,
            for line in open(paths[i], 'rb'):
                tp = [0] * (len(word2id) + 4)
                for w in jieba.cut(line):
                    for key in word2id:
                        if w in key:
                            tp[word2id[key]] += 1
                    for key in word2id_cat:
                        if w in key:
                            tp[-(word2id_cat[key] + 2)] += 1
                tp[-1] = i
                data.append(tp)
        data = np.array(data)
        return data

    elif clabel == 2:
        word2id = {}
        word2id_cat = {}
        path = 'D:\CodeProject\PythonProject\\nlp_zhxg\data\combin_2\\'
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

        for i in range(4, 7):
            for line in open(keyword_path + str(i) + ".csv"):
                for w in line.split():
                    word2id[w.strip().decode("utf-8")] = len(word2id)

        for i in range(4, 7):
            for line in open(keyword_path + str(i) + ".csv"):
                for w in line.split():
                    word2id_cat[w.strip().decode("utf-8")] = i - 4

        data = []
        paths = [path + "yjqx.txt", path + "khwl.txt", path + "jz.txt", path + "others.txt"]
        for i in range(4):
            # print i,
            for line in open(paths[i], 'rb'):
                tp = [0] * (len(word2id) + 4)
                for w in jieba.cut(line):
                    for key in word2id:
                        if w in key:
                            tp[word2id[key]] += 1
                    for key in word2id_cat:
                        if w in key:
                            tp[-(word2id_cat[key] + 2)] += 1
                tp[-1] = i
                data.append(tp)
        data = np.array(data)
        return data

    elif clabel == 3:
        word2id = {}
        word2id_cat = {}
        path = 'D:\CodeProject\PythonProject\\nlp_zhxg\data\combin_3\\'
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

        for i in range(7, 10):
            for line in open(keyword_path + str(i) + ".csv"):
                for w in line.split():
                    word2id[w.strip().decode("utf-8")] = len(word2id)

        for i in range(7, 10):
            for line in open(keyword_path + str(i) + ".csv"):
                for w in line.split():
                    word2id_cat[w.strip().decode("utf-8")] = i - 7

        data = []
        paths = [path + "dcxz.txt", path + "kdts.txt", path + "sjhx.txt", path + "others.txt"]
        for i in range(4):
            # print i,
            for line in open(paths[i], 'rb'):
                tp = [0] * (len(word2id) + 4)
                for w in jieba.cut(line):
                    for key in word2id:
                        if w in key:
                            tp[word2id[key]] += 1
                    for key in word2id_cat:
                        if w in key:
                            tp[-(word2id_cat[key] + 2)] += 1
                tp[-1] = i
                data.append(tp)
        data = np.array(data)
        return data

    elif clabel == 4:
        word2id = {}
        word2id_cat = {}
        path = 'D:\CodeProject\PythonProject\\nlp_zhxg\data\combin_4\\'
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

        for i in range(10, 13):
            for line in open(keyword_path + str(i) + ".csv"):
                for w in line.split():
                    word2id[w.strip().decode("utf-8")] = len(word2id)

        for i in range(10, 13):
            for line in open(keyword_path + str(i) + ".csv"):
                for w in line.split():
                    word2id_cat[w.strip().decode("utf-8")] = i - 10

        data = []
        paths = [path + "kdgz.txt", path + "kdwsm.txt", path + "ydwlxhc.txt", path + "others.txt"]
        for i in range(4):
            # print i,
            for line in open(paths[i], 'rb'):
                tp = [0] * (len(word2id) + 4)
                for w in jieba.cut(line):
                    for key in word2id:
                        if w in key:
                            tp[word2id[key]] += 1
                    for key in word2id_cat:
                        if w in key:
                            tp[-(word2id_cat[key] + 2)] += 1
                tp[-1] = i
                data.append(tp)
        data = np.array(data)
        return data

    elif clabel == 5:
        word2id = {}
        word2id_cat = {}
        path = 'D:\CodeProject\PythonProject\\nlp_zhxg\data\combin_5\\'
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

        for i in range(13, 16):
            for line in open(keyword_path + str(i) + ".csv"):
                for w in line.split():
                    word2id[w.strip().decode("utf-8")] = len(word2id)

        for i in range(13, 16):
            for line in open(keyword_path + str(i) + ".csv"):
                for w in line.split():
                    word2id_cat[w.strip().decode("utf-8")] = i - 13

        data = []
        paths = [path + "fscpjb.txt", path + "dkxz.txt", path + "dtvxz.txt", path + "others.txt"]
        for i in range(4):
            # print i,
            for line in open(paths[i], 'rb'):
                tp = [0] * (len(word2id) + 4)
                for w in jieba.cut(line):
                    for key in word2id:
                        if w in key:
                            tp[word2id[key]] += 1
                    for key in word2id_cat:
                        if w in key:
                            tp[-(word2id_cat[key] + 2)] += 1
                tp[-1] = i
                data.append(tp)
        data = np.array(data)
        return data

    elif clabel == 6:
        word2id = {}
        word2id_cat = {}
        path = 'D:\CodeProject\PythonProject\\nlp_zhxg\data\combin_6\\'
        for line in open(path + "keywords_single_250.txt"):
            for w in line.split():
                word2id[w.strip().decode("utf-8")] = len(word2id)
        ct = 0
        counts = [0, 0, 0, 0, 0]

        for line in open(path + "keywords_single_250.txt"):
            for w in line.split():
                word2id_cat[w.strip().decode("utf-8")] = ct
                counts[ct] += 1
            ct += 1

        for i in range(16, 21):
            for line in open(keyword_path + str(i) + ".csv"):
                for w in line.split():
                    word2id[w.strip().decode("utf-8")] = len(word2id)

        for i in range(16, 21):
            for line in open(keyword_path + str(i) + ".csv"):
                for w in line.split():
                    word2id_cat[w.strip().decode("utf-8")] = i - 16

        data = []
        paths = [path + "rhlw.txt", path + "lw.txt", path + "rhcf.txt", path + "znzw.txt",path + 'kdwzy.txt', path + "others.txt"]
        for i in range(6):
            # print i,
            for line in open(paths[i], 'rb'):
                tp = [0] * (len(word2id) + 6)
                for w in jieba.cut(line):
                    for key in word2id:
                        if w in key:
                            tp[word2id[key]] += 1
                    for key in word2id_cat:
                        if w in key:
                            tp[-(word2id_cat[key] + 2)] += 1
                tp[-1] = i
                data.append(tp)
        data = np.array(data)
        return data



